"""Test the novice UI's paired evaluation independently of Streamlit."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import g8_core as c


class TestG8Evidence(unittest.TestCase):
    def test_surface_counts_and_scope(self):
        t=c.surface_comparison()
        self.assertEqual(len(t),3600)
        noise=c.surface_stats(t[t.sigma>0])
        self.assertEqual((noise['before_errors'],noise['after_errors'],noise['fixed'],noise['new_errors']),
                         (460,50,418,8))
        clean=c.surface_stats(t[t.sigma==0])
        self.assertEqual(clean['n'],360)
        self.assertEqual((clean['before_errors'],clean['after_errors']),(1,3))

    def test_binary_all_conditions(self):
        data=c.binary_comparison();self.assertEqual(len(data),3262)
        s=c.binary_stats(data)
        self.assertEqual((s['K0']['fn'],s['K0']['fp']),(48,601))
        self.assertEqual((s['K1']['fn'],s['K1']['fp']),(76,263))
        self.assertEqual(int(data.change.eq('fixed').sum()),415)
        self.assertEqual(int(data.change.eq('new_error').sum()),105)
        self.assertEqual(int(data.groupby('condition').size().sum()),3262)

    def test_binary_detects_misaligned_truth(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('K0','K1'):
                df=pd.read_csv(c.BIN/f'{name}_validation_diagnostics.csv')
                if name=='K1':
                    df.loc[0,'true_label']='defect' if df.loc[0,'true_label']=='normal' else 'normal'
                df.to_csv(root/f'{name}_validation_diagnostics.csv',index=False)
            with self.assertRaises(ValueError):c.binary_comparison(root)

    def test_binary_detects_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('K0','K1'):
                df=pd.read_csv(c.BIN/f'{name}_validation_diagnostics.csv')
                if name=='K1':df=pd.concat([df,df.iloc[:1]],ignore_index=True)
                df.to_csv(root/f'{name}_validation_diagnostics.csv',index=False)
            with self.assertRaises(ValueError):c.binary_comparison(root)

    def test_slag_same_windows(self):
        frames,s=c.slag_comparison()
        self.assertEqual(len(frames),3)
        self.assertTrue(all(len(v)==240 for v in frames.values()))
        table=s.set_index('model')
        g=table.loc['G6-B ExtraTrees']; deep=table.loc['G6 Deep MLP']
        self.assertEqual((g.TP,g.FN,g.FP,g.TN),(86,74,5,75))
        self.assertEqual((deep.TP,deep.FN,deep.FP,deep.TN),(49,111,7,73))
        changes=c.changes_for_slag(frames,'G6-B ExtraTrees','G6 Deep MLP')
        self.assertEqual(int(changes.change.eq('fixed').sum()),10)
        self.assertEqual(int(changes.change.eq('new_error').sum()),49)

    def test_slag_disallows_same_model(self):
        frames,_=c.slag_comparison()
        with self.assertRaises(ValueError):c.changes_for_slag(frames,'G6-B ExtraTrees','G6-B ExtraTrees')


if __name__=='__main__':unittest.main()
