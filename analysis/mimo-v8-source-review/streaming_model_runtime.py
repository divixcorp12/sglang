"""MiMo text adapter for upstream oneshot/sequential and DiskCache.

Packed source tensors are registered meta buffers. Upstream owns staging,
onloading, activation caching, hooks and offloading. Compute defaults to decoded BF16; an explicit SM120 runtime profile is available.
"""
import json,struct
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig,PreTrainedModel,AutoConfig
from compressed_tensors.offload import offload_module
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.cache.base import OffloadCache
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from mimo_compressor.adapter import MiMoGate,MiMoMoE
from mimo_compressor.decoder import source_core,load_config,decode_fp8_blocks,decode_tp_qkv,qkv_shard_sizes,position_inputs
from mimo_compressor.packed import decode_mxfp4

DTYPES={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16,'U8':torch.uint8,'F8_E4M3':torch.float8_e4m3fn}

def verify_meta_release(model):
 count=0
 for module in model.modules():
  for mapping in (module._parameters,module._buffers):
   if isinstance(mapping,OffloadCache):
    for value in mapping.offloaded_values.values():
     if value is not None:
      if value.device.type!='meta':raise ValueError('Source tensor remained resident after sequential offload')
      count+=1
 return count

class SourceRegistry:
 def __init__(self,root,offload_dir,precision="decoded_bf16"):
  from mimo_compressor.runtime_precision import PROFILES
  if precision not in PROFILES:raise ValueError("Unknown precision profile")
  self.precision=precision
  self.root=Path(root).resolve();self.offload_dir=Path(offload_dir).resolve();self.offload_dir.mkdir(parents=True,exist_ok=True)
  if self.offload_dir==self.root or self.root in self.offload_dir.parents:raise ValueError('Offload output must be outside source model')
  self.index=json.load(open(self.root/'model.safetensors.index.json'))['weight_map'];self.headers={};self.registered_bytes=0
 def tensor(self,key,parameter=False):
  path=(self.root/self.index[key]).resolve()
  if path.parent!=self.root:raise ValueError('Shard path escapes source')
  if path not in self.headers:
   with path.open('rb') as f:
    size=struct.unpack('<Q',f.read(8))[0]
    if size>128*(1<<20):raise ValueError('Oversized tensor header')
    self.headers[path]=json.loads(f.read(size))
  info=self.headers[path][key];value=torch.empty(info['shape'],device='meta',dtype=DTYPES[info['dtype']])
  if parameter:value=nn.Parameter(value,requires_grad=False)
  DiskCache.index[value]={'safetensors_file':str(path),'weight_name':key,'dtype':str(value.dtype).removeprefix('torch.')}
  self.registered_bytes+=value.numel()*value.element_size();return value
 def dispatch(self,model):
  for module in list(model.modules()):
   if module._parameters or module._buffers:
    offload_module(module,onload_device='cuda',offload_device='disk',offload_dir=str(self.offload_dir),read_only=True)

class SourceLinear(nn.Module):
 def __init__(self,registry,key,qkv_sizes=None,fp4=False):
  super().__init__();self.qkv_sizes=qkv_sizes;self.fp4=fp4;self.precision=registry.precision
  self.register_buffer('weight',registry.tensor(key+'.weight'))
  scale_key=key+('.weight_scale' if fp4 else '.weight_scale_inv')
  self.register_buffer('scale',registry.tensor(scale_key) if scale_key in registry.index else None)
 def forward(self,x):
  if self.precision=="sglang_sm120" and not self.fp4 and self.scale is not None:
   from mimo_compressor.runtime_precision import fp8_linear,requantize_fp8_weight
   if self.qkv_sizes is not None:
    weight,scale=requantize_fp8_weight(decode_tp_qkv(self.weight,self.scale,self.qkv_sizes,dtype=torch.bfloat16))
   else:weight,scale=self.weight,self.scale
   return fp8_linear(x,weight,scale)
  if self.fp4:w=decode_mxfp4(self.weight,self.scale,x.dtype)
  elif self.scale is None:w=self.weight.to(x.dtype)
  elif self.qkv_sizes is not None:w=decode_tp_qkv(self.weight,self.scale,self.qkv_sizes,dtype=x.dtype)
  else:w=decode_fp8_blocks(self.weight,self.scale,dtype=x.dtype)
  return F.linear(x,w)

class SourceExpert(ExpertMLP):
 def __init__(self,registry,layer,expert):
  nn.Module.__init__(self);self.precision=registry.precision;key=f'model.layers.{layer}.mlp.experts.{expert}'
  for name in ('gate_proj','up_proj','down_proj'):setattr(self,name,SourceLinear(registry,key+'.'+name,fp4=True))
 def copy_from_experts_module(self,*args):raise NotImplementedError('Registered packed source expert')
 def forward(self,x):
  if self.precision=="sglang_sm120":
   from mimo_compressor.runtime_precision import expert
   return expert(x,self.gate_proj.weight,self.up_proj.weight,self.down_proj.weight,
    self.gate_proj.scale,self.up_proj.scale,self.down_proj.scale)
  return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))

class MiMoAttentionStage(nn.Module):
 def __init__(self,norm,attention,cfg,index):
  super().__init__();self.input_layernorm=norm;self.self_attn=attention;self.cfg=cfg;self.index=index
 def forward(self,hidden_states):
  normalized=self.input_layernorm(hidden_states)
  output,_=self.self_attn(normalized,**position_inputs(self.cfg,self.index,normalized))
  return hidden_states+output

class MiMoFeedForwardStage(nn.Module):
 def __init__(self,norm,mlp):
  super().__init__();self.post_attention_layernorm=norm;self.mlp=mlp
 def forward(self,hidden_states):return hidden_states+self.mlp(self.post_attention_layernorm(hidden_states))

class MiMoSequentialLayer(nn.Module):
 def __init__(self,registry,cfg,index,split=False):
  super().__init__();self.cfg=cfg;self.index=index;core=source_core(registry.root);prefix=f'model.layers.{index}'
  with torch.device('meta'):
   layer=core['MiMoV2DecoderLayer'].__new__(core['MiMoV2DecoderLayer']);nn.Module.__init__(layer)
   swa=bool(cfg.hybrid_layer_pattern[index]);layer.attention_type='sliding_window_attention' if swa else 'full_attention'
   layer.self_attn=core['MiMoV2Attention'](cfg,swa,index,projection_layout='fused_qkv')
   for name in ('input_layernorm','post_attention_layernorm'):
    norm=core['MiMoV2RMSNorm'](cfg.hidden_size,eps=cfg.layernorm_epsilon);norm.weight=registry.tensor(prefix+'.'+name+'.weight',parameter=True);setattr(layer,name,norm)
   layer.self_attn.qkv_proj=SourceLinear(registry,prefix+'.self_attn.qkv_proj',qkv_shard_sizes(cfg,index))
   layer.self_attn.o_proj=SourceLinear(registry,prefix+'.self_attn.o_proj')
   if layer.self_attn.attention_sink_bias is not None:layer.self_attn.attention_sink_bias=registry.tensor(prefix+'.self_attn.attention_sink_bias',parameter=True)
   if cfg.moe_layer_freq[index]:
    gate_weight=registry.tensor(prefix+'.mlp.gate.weight',parameter=True)
    gate=MiMoGate(cfg,gate_weight,registry.tensor(prefix+'.mlp.gate.e_score_correction_bias'))
    # MiMoGate wraps its input in a new Parameter; register that exact object.
    DiskCache.index[gate.weight]=DiskCache.index[gate_weight]
    layer.mlp=MiMoMoE(gate,[SourceExpert(registry,index,e) for e in range(cfg.n_routed_experts)])
   else:
    layer.mlp=core['MiMoV2MLP'](cfg)
    for name in ('gate_proj','up_proj','down_proj'):setattr(layer.mlp,name,SourceLinear(registry,prefix+'.mlp.'+name))
  self.split=split
  if split:
   self.attention_stage=MiMoAttentionStage(layer.input_layernorm,layer.self_attn,cfg,index)
   self.feed_forward_stage=MiMoFeedForwardStage(layer.post_attention_layernorm,layer.mlp)
  else:self.decoder=layer
 def forward(self,hidden_states):
  if self.split:return self.feed_forward_stage(self.attention_stage(hidden_states))
  return self.decoder(hidden_states,**position_inputs(self.cfg,self.index,hidden_states))

class MiMoDiskConfig(PretrainedConfig):
 model_type='mimo_disk_adapter'

AutoConfig.register('mimo_disk_adapter',MiMoDiskConfig)

class MiMoDiskModel(PreTrainedModel):
 config_class=MiMoDiskConfig
 _no_split_modules=['MiMoSequentialLayer']
 def __init__(self,registry,layers,config_dir,split=False):
  cfg=load_config(registry.root,'mimo_chunked');data=vars(cfg).copy();data.pop('quantization_config',None);data.pop('model_type',None);data.pop('architectures',None);data['_name_or_path']=str(config_dir);data['num_hidden_layers']=layers
  data['observer_precision_profile']=registry.precision
  config=MiMoDiskConfig(**data);super().__init__(config)
  # Adapter config identifies the compute profile; original packed bytes stay in source buffers.
  config.save_pretrained(config_dir);self.source_model_path=str(registry.root)
  with torch.device('meta'):self.embed_tokens=nn.Embedding(cfg.vocab_size,cfg.hidden_size)
  self.embed_tokens.weight=registry.tensor('model.embed_tokens.weight',parameter=True)
  self.layers=nn.ModuleList([MiMoSequentialLayer(registry,cfg,i,split=split) for i in range(layers)])
  registry.dispatch(self)
 def get_input_embeddings(self):return self.embed_tokens
 def get_output_embeddings(self):return None
 def forward(self,input_ids):
  hidden_states=self.embed_tokens(input_ids)
  for layer in self.layers:hidden_states=layer(hidden_states)
  return hidden_states
