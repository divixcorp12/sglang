import copy
import unittest
from finance_union_v9 import merge_rows


class UnionIdentityTests(unittest.TestCase):
    def row(self):
        return dict(sample_id='sample', source_record_sha256='raw', gap_category='risk',
            partition='primary', messages=[dict(role='user', content='task')],
            token_ids=[1, 2], token_sha256='tokens', preparation_policy_sha256='old')

    def test_duplicate_keeps_complete_original_and_records_discovery_policy(self):
        old = self.row()
        new = dict(old, preparation_policy_sha256='extension')
        merged, duplicates = merge_rows([old], [new])
        self.assertEqual(merged, [old])
        self.assertEqual(duplicates[0]['preparation_policy_sha256'], 'extension')

    def test_conflicting_duplicate_fields_are_rejected(self):
        for field, value in [('messages', []), ('token_ids', [9]),
                ('token_sha256', 'changed'), ('partition', 'repeat')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                merge_rows([self.row()], [dict(self.row(), **{field: value})])

    def test_same_sample_cannot_change_raw_source_across_categories(self):
        with self.assertRaises(ValueError):
            merge_rows([self.row()], [dict(self.row(), gap_category='other',
                source_record_sha256='different')])

    def test_category_alias_retained_without_modifying_input(self):
        old = self.row()
        before = copy.deepcopy(old)
        merged, duplicates = merge_rows([old], [dict(old, gap_category='empirical')])
        self.assertEqual(len(merged), 2)
        self.assertEqual(duplicates, [])
        merged[0]['messages'][0]['content'] = 'mutated copy'
        self.assertEqual(old, before)


if __name__ == '__main__':
    unittest.main()
