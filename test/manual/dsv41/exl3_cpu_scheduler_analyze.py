"""Join bounded CPU worker records to an Nsight SQLite scheduler export on divix01."""
import argparse
import bisect
import collections
import json
from pathlib import Path
import sqlite3


def overlap(a, b, c, d):
    return max(0, min(b, d) - max(a, c))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    root = args.directory
    arm = json.loads((root / "arm.json").read_text())
    jobs = arm["jobs"]
    starts = [j['start_ns'] for j in jobs]
    frames = collections.defaultdict(list)
    footer = None
    for f in root.glob("worker-phases.*.jsonl"):
        for line in f.open():
            r = json.loads(line)
            if 'forward' in r:
                frames[r['forward']].append(r)
            elif r.get('footer'):
                footer = r
    assert footer and footer['dropped_forwards'] == 0
    tids = sorted({r['tid'] for rs in frames.values() for r in rs})
    connection = sqlite3.connect(root / "scheduler.sqlite")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA temp_store=FILE")
    epoch, system = connection.execute("SELECT utcEpochNs,systemClockNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()
    sched_range = connection.execute("SELECT min(start),max(start) FROM SCHED_EVENTS").fetchone()
    relative = [jobs[0]['start_ns'] - system, jobs[-1]['end_ns'] - system]
    assert sched_range[0] < relative[0] < relative[1] < sched_range[1], (sched_range, relative)
    types = dict(connection.execute("SELECT s.value,t.typeId FROM GENERIC_EVENT_TYPES t JOIN StringIds s ON s.id=t.nameId"))
    # The ftrace field schema is verified rather than assuming a kernel version's offsets.
    fields = collections.defaultdict(dict)
    for type_id, idx, name in connection.execute("SELECT f.typeId,f.fieldIdx,s.value FROM GENERIC_EVENT_TYPE_FIELDS f JOIN StringIds s ON s.id=f.fieldNameId"):
        fields[type_id][name] = idx
    tid_sql = ','.join(map(str, tids))
    low, high = relative[0] - 50_000_000, relative[1] + 50_000_000

    def extract(type_name, names, target_names):
        type_id = types[type_name]
        indexes = [fields[type_id][name] for name in names]
        values = [f"max(CASE WHEN d.fieldIdx={idx} THEN coalesce(d.intVal,d.uintVal) END)" for idx in indexes]
        having = ' OR '.join(values[names.index(name)] + f" IN ({tid_sql})" for name in target_names)
        sql = "SELECT e.timestamp,e.genericEventId," + ','.join(values) + " FROM GENERIC_EVENTS e JOIN GENERIC_EVENT_DATA d ON d.genericEventId=e.genericEventId WHERE e.typeId=? AND e.timestamp BETWEEN ? AND ? AND d.fieldIdx IN (" + ','.join(map(str,indexes)) + ") GROUP BY e.genericEventId HAVING " + having + " ORDER BY e.timestamp"
        rows = connection.execute(sql, (type_id, low, high)).fetchall()
        return [dict(ns=system+row[0], event_id=row[1], **dict(zip(names,row[2:]))) for row in rows]

    switches = extract('sched:sched_switch', ['prev_pid','prev_state','next_pid'], ['prev_pid','next_pid'])
    wakes = extract('sched:sched_wakeup', ['common_pid','pid','target_cpu'], ['pid'])
    bytid = collections.defaultdict(list)
    for w in wakes:
        bytid[w['pid']].append(w)
    intervals = collections.defaultdict(list)
    pending = {}
    for e in switches:
        if e['prev_pid'] in tids:
            pending[e['prev_pid']] = e
        if e['next_pid'] in tids:
            tid = e['next_pid']
            out = pending.pop(tid, None)
            if out is None:
                continue
            wake = next((w for w in bytid[tid] if out['ns'] <= w['ns'] <= e['ns']), None)
            # State 0 is runnable. Other Linux prev_state values denote non-running states.
            runnable = out['ns'] if out['prev_state'] == 0 else wake['ns'] if wake else None
            intervals[tid].append({'out_ns':out['ns'], 'in_ns':e['ns'], 'state':out['prev_state'],
                                   'wake_ns':wake['ns'] if wake else None, 'runnable_ns':runnable,
                                   'next_pid_on_out':out['next_pid'], 'waker_tid':wake['common_pid'] if wake else None})

    def describe(tid, lo, hi):
        d = dict(offcpu_ms=0., sleep_ms=0., runnable_ms=0., unresolved_ms=0., intervals=[])
        for v in intervals[tid]:
            n = overlap(lo, hi, v['out_ns'], v['in_ns'])
            if not n:
                continue
            d['offcpu_ms'] += n / 1e6
            if v['runnable_ns'] is None:
                d['unresolved_ms'] += n / 1e6
            else:
                d['sleep_ms'] += overlap(lo, hi, v['out_ns'], v['runnable_ns']) / 1e6
                d['runnable_ms'] += overlap(lo, hi, v['runnable_ns'], v['in_ns']) / 1e6
            d['intervals'].append(v)
        return d

    joined = []
    for serial, rs in frames.items():
        assert len(rs) == 50 and len({(r['worker'],r['phase']) for r in rs}) == 50
        first = rs[0]
        i = bisect.bisect_right(starts,first['forward_begin']) - 1
        if i < 0 or jobs[i]['end_ns'] < first['forward_end']:
            continue
        last_enter = max(rs,key=lambda r:r['enter'])
        record = {'forward':serial,'job':jobs[i], 'entry':{'worker':last_enter['worker'],'tid':last_enter['tid'],
                   'delay_ms':(last_enter['enter'] - first['team_begin'])/1e6,
                   'scheduler':describe(last_enter['tid'],first['team_begin'],last_enter['enter'])},'phases':[]}
        for phase in (0,1,2,3):
            ps = [r for r in rs if r['phase']==phase]
            work_done = max(r['work_end'] for r in ps)
            last = max(ps,key=lambda r:r['end'])
            record['phases'].append({'phase':phase,'last_exit_worker':last['worker'],'last_exit_tid':last['tid'],
               'all_work_done_ns':work_done,'last_exit_ns':last['end'],'release_tail_ms':(last['end']-work_done)/1e6,
               'scheduler_after_all_work':describe(last['tid'],work_done,last['end'])})
        joined.append(record)
    assert len(joined) == len(jobs) == 600
    barrier = sorted([{'forward':f['forward'],'job':f['job'],**p} for f in joined for p in f['phases']],key=lambda p:p['release_tail_ms'],reverse=True)
    evidence={'system_clock_ns':system,'utc_epoch_ns':epoch,'scheduler_range_ns':list(sched_range),'job_relative_range_ns':relative,
              'coverage_valid':True,'timed_jobs':len(joined),'worker_footer':footer,'target_tids':tids,
              'switch_count':len(switches),'wake_count':len(wakes),
              'diagnostics':connection.execute("SELECT timestamp,severity,text FROM DIAGNOSTIC_EVENT").fetchall(),
              'top_barriers':barrier[:20], 'top_entries':sorted(joined,key=lambda f:f['entry']['delay_ms'],reverse=True)[:10]}
    (root/'scheduler-analysis.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (root/'scheduler-events.json').write_text(json.dumps({'switches':switches,'wakes':wakes},indent=2)+'\n')
    print('VALID',len(joined),'jobs',len(switches),'switches',len(wakes),'wakes')
    for b in barrier[:6]:
        print('BARRIER',b['forward'],b['phase'],b['release_tail_ms'],{k:v for k,v in b['scheduler_after_all_work'].items() if k!='intervals'})
    for f in evidence['top_entries'][:3]:
        print('ENTRY',f['forward'],f['entry'])


if __name__ == '__main__':
    main()
