"""G8.2 coverage: all models, identical test records, cohort guard, metrics."""
import unittest
import pandas as pd
from src import g8_catalog as c


class TestG82ModelCatalog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surf=c.surface_records('validation')
        cls.test=c.surface_records('test')
        cls.slag=c.slag_records()

    def test_surface_four_models_same_raw_inputs(self):
        self.assertEqual(set(self.surf),set(c.SURFACE_MODELS))
        self.assertEqual({len(x) for x in self.surf.values()},{3600})
        self.assertEqual(len(c.surface_summary(self.surf)),4)
        for a in c.SURFACE_MODELS:
            for b in c.SURFACE_MODELS:
                if a!=b:self.assertEqual(len(c.surface_pair(a,b,self.surf)),3600)

    def test_surface_test_is_not_validation(self):
        self.assertEqual(set(self.test),{'V0','G1'})
        df=c.surface_summary(self.test).set_index('모델')
        self.assertEqual(int(df.loc['V0','잡음 오분류(회)']),460)
        self.assertEqual(int(df.loc['G1','잡음 오분류(회)']),50)
        pair=c.surface_pair('V0','G1',self.test)
        self.assertEqual(int(pair[pair.sigma>0].change.eq('fixed').sum()),418)
        self.assertEqual(int(pair[pair.sigma>0].change.eq('new_error').sum()),8)

    def test_surface_different_raw_hash_rejected(self):
        a=pd.DataFrame({'id':[1], 'truth':['a'], 'hash':['x']})
        b=pd.DataFrame({'id':[1], 'truth':['a'], 'hash':['y']})
        with self.assertRaises(ValueError):c._exactly_same_keys({'a':a,'b':b},['id'],'truth','hash')

    def test_surface_duplicate_rejected(self):
        a=pd.DataFrame({'id':[1,1], 'truth':['a','a']})
        with self.assertRaises(ValueError):c._exactly_same_keys({'a':a},['id'],'truth')

    def test_all_eight_slag_models_registered(self):
        found={k for frames in self.slag.values() for k in frames}
        self.assertEqual(found,set(c.SLAG_MODELS))
        self.assertEqual(len(c.slag_summary(self.slag)),9)  # J compared on 2 distinct cohorts

    def test_early_and_e_reference_metrics(self):
        early=self.slag['기초 진동 · E/B/S 480구간']
        mid=self.slag['E 기준 진동 · E/B/S 400구간']
        self.assertEqual({len(t) for t in early.values()},{480})
        self.assertEqual({len(t) for t in mid.values()},{400})
        self.assertEqual((c.slag_measures(mid['J'])['TP'],c.slag_measures(mid['J'])['FP']),(49,41))
        self.assertEqual((c.slag_measures(mid['G3R'])['TP'],c.slag_measures(mid['G3R'])['FP']),(64,29))

    def test_b_reference_metrics(self):
        late=self.slag['B 기준 진동 · B/S 240구간']
        self.assertEqual((c.slag_measures(late['G5-B'])['TP'],c.slag_measures(late['G5-B'])['FP']),(81,5))
        self.assertEqual((c.slag_measures(late['G6-B'])['TP'],c.slag_measures(late['G6-B'])['FP']),(86,5))
        self.assertEqual((c.slag_measures(late['G6 Deep'])['TP'],c.slag_measures(late['G6 Deep'])['FP']),(49,7))
        paired=c.slag_pair('B 기준 진동 · B/S 240구간','G5-B','G6-B',self.slag)
        self.assertEqual((int(paired.change.eq('fixed').sum()),int(paired.change.eq('new_error').sum())),(20,15))

    def test_cross_cohort_not_compared(self):
        with self.assertRaises(ValueError):c.slag_pair('B 기준 진동 · B/S 240구간','G3R','G6-B',self.slag)
        with self.assertRaises(ValueError):c.slag_pair('기초 진동 · E/B/S 480구간','J','J',self.slag)

    def test_key_pair_modified_truth_rejected(self):
        p=self.slag['B 기준 진동 · B/S 240구간']
        bad={k:v.copy() for k,v in p.items()}
        bad['G6-B'].loc[0,'truth']=1-bad['G6-B'].loc[0,'truth']
        copy=dict(self.slag);copy['B 기준 진동 · B/S 240구간']=bad
        with self.assertRaises(ValueError):c.slag_pair('B 기준 진동 · B/S 240구간','G5-B','G6-B',copy)


if __name__=='__main__':unittest.main()
