import copy
import unittest
from balanced_finance_draft_v9 import allocate


class ContentAllocationTests(unittest.TestCase):
    def row(self, sid, part='primary', tokens=18000, **kw):
        return dict(sample_id=sid, source_record_sha256='raw-' + sid,
            split_group='research_question:' + sid, partition=part,
            valid_tokens=tokens, gap_category='risk', stratum='research',
            target_domain='finance', messages=[dict(role='user', content=sid)],
            token_ids=[1, 2], **kw)

    def run_pool(self, pool, owners=None, validate=None, locked=None):
        return allocate(pool, locked or [], owners or {}, {'risk': 1}, 'fixed-seed',
            lambda n: 'long' if n >= 16384 else 'short', validate or (lambda rows, own: None))

    def test_same_side_unowned_groups_can_be_allocated_without_mutating_history(self):
        pool = [self.row('a'), self.row('b')]
        before = copy.deepcopy(pool)
        selected, owners, ledger, gaps = self.run_pool(pool)
        self.assertFalse(gaps)
        self.assertEqual({r['partition'] for r in selected}, {'primary', 'repeat'})
        self.assertEqual(pool, before)
        for r in selected:
            original = next(p for p in pool if p['sample_id'] == r['sample_id'])
            self.assertEqual({k: v for k, v in r.items() if k != 'partition'},
                {k: v for k, v in original.items() if k != 'partition'})

    def test_parent_ownership_never_moves(self):
        pool = [self.row('a'), self.row('b')]
        owners = {r['split_group']: 'primary' for r in pool}
        self.assertTrue(self.run_pool(pool, owners=owners)[3])

    def test_lineage_constraints_reject_infeasible_orientation(self):
        def check(rows, owners):
            if next(r for r in rows if r['sample_id'] == 'a')['partition'] != 'primary':
                raise ValueError('opposite parent paper')
        selected, _, _, gaps = self.run_pool([self.row('a'), self.row('b')], validate=check)
        self.assertFalse(gaps)
        self.assertEqual(next(r for r in selected if r['sample_id'] == 'a')['partition'], 'primary')

    def test_different_context_buckets_remain_unfilled(self):
        self.assertTrue(self.run_pool([self.row('a', tokens=15000), self.row('b')])[3])

    def test_length_ratio_not_relaxed(self):
        self.assertTrue(self.run_pool([self.row('a'), self.row('b', tokens=30000)])[3])

    def test_duplicate_source_cannot_form_pair(self):
        a, b = self.row('a'), self.row('b')
        b['source_record_sha256'] = a['source_record_sha256']
        self.assertTrue(self.run_pool([a, b])[3])

    def test_enumeration_order_does_not_select_another_pair(self):
        pool = [self.row('a'), self.row('b'), self.row('c')]
        self.assertEqual(self.run_pool(pool), self.run_pool(list(reversed(pool))))

    def test_locked_histories_and_partitions_remain_exact(self):
        locked = [self.row('old-primary'), self.row('old-repeat', 'repeat')]
        self.assertEqual(self.run_pool([self.row('new')], locked=locked)[0], locked)

    def test_existing_mixed_draft_imbalance_cannot_be_repaired_by_moves(self):
        with self.assertRaises(ValueError):
            self.run_pool([self.row('new')], locked=[self.row('old')])


if __name__ == '__main__':
    unittest.main()
