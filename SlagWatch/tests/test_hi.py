import hashlib
import unittest
import numpy as np
import pandas as pd
import joblib
from streamlit.testing.v1 import AppTest
from src.data import load_dataset
from src.hi_study import ROOT,OUT,DEST,read_config,select_cut,summarize,make_wave
from src.hi_models import robust36,predict_signal


class TestHI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=read_config()
        cls.p=pd.read_csv(DEST/"predictions.csv",float_precision="round_trip")
        cls.meta,cls.w,cls.man=load_dataset(ROOT/"data/raw")

    def test_cut_objective_and_training_only(self):
        sel=select_cut(np.array([0,0,0,0,1,1]),np.array([.2,.3,.7,.8,.72,.85]),.25)
        self.assertEqual((sel["threshold"],sel["tp"],sel["fp"]),(.72,2,1))
        votes=pd.read_csv(DEST/"inner_oof_clean.csv",float_precision="round_trip")
        cuts=pd.read_csv(DEST/"thresholds.csv",float_precision="round_trip")
        for f in self.c["folds"]:
            train=set(f['train']);test=set(f['test'])
            self.assertFalse(train&test)
            self.assertEqual(len(train),12);self.assertEqual(len(test),4)
            coverage=[]
            for s in f['inner']:
                self.assertEqual(set(s['train'])|set(s['val']),train)
                self.assertFalse(set(s['train'])&set(s['val']))
                coverage+=s["val"]
            self.assertEqual(sorted(coverage),sorted(train))
            for m in ['A1','H','I']:
                v=votes[(votes.outer_fold==f['fold_id'])&(votes.model_id==m)]
                self.assertEqual(len(v),360);self.assertEqual(set(v.group_id),train)
                t=select_cut(v.y_true.to_numpy(),v.score.to_numpy(),.1)
                given=cuts[(cuts.fold_id==f['fold_id'])&(cuts.model_id==m)].iloc[0]
                self.assertEqual(t['threshold'],given.threshold)
                self.assertLessEqual(t['fp'],24)
                eval_rows=self.p[(self.p.fold_id==f['fold_id'])&(self.p.model_id==m)]
                self.assertEqual(set(eval_rows.group_id),test)
                self.assertEqual(list(eval_rows.threshold.unique()),[given.threshold])

    def test_population_and_v4_regression(self):
        p=self.p
        keys=['file_id','window_idx','scenario']
        self.assertEqual(len(p),4*5*480)
        self.assertFalse(p.duplicated(keys+['model_id']).any())
        self.assertTrue((p.groupby(keys).input_sha256.nunique()==1).all())
        hashes=p[p.scenario=='clean'].drop_duplicates('input_sha256')
        self.assertEqual(len(hashes),480)
        old=pd.concat([pd.read_csv(OUT/'paired_scores.csv'),pd.read_csv(OUT/'multi_scores.csv')])
        v=old[(old.family=='spectral')&(old.variant=='A')]
        j=p[p.model_id=='A0'].merge(v,on=keys,validate='one_to_one')
        self.assertEqual(len(j),2400)
        np.testing.assert_allclose(j.score_x,j.score_y,atol=1e-12,rtol=0)
        self.assertTrue(np.array_equal(j.prediction,(j.score_y>=.5)))
        summary=pd.read_csv(DEST/'summary.csv')
        for (m,s),g in p.groupby(['model_id','scenario']):
            t=summarize(g)
            row=summary[(summary.model_id==m)&(summary.scenario==s)].iloc[0]
            for key,num in t.items():self.assertAlmostEqual(num,row[key])
            self.assertEqual(t['tp']+t['fn'],160);self.assertEqual(t['fp']+t['tn'],320)
            self.assertEqual(t['fp_E']+t['fp_B'],t['fp'])

    def test_replay_and_invalid_input(self):
        config=self.c['features_I']
        zero=robust36(np.zeros((3200,3)),config)
        self.assertEqual(len(zero),36);self.assertTrue(np.isfinite(zero).all())
        for f in range(4):
            b=joblib.load(DEST/f'models/fold_{f}.joblib')
            groups=b['train_groups']
            ix=self.meta.index[self.meta.group_id.isin(groups)]
            np.testing.assert_allclose(b['scale'],np.median([self.w[i].std(axis=0) for i in ix],axis=0),atol=1e-12)
            train_x=np.stack([robust36(self.w[i],config) for i in ix])
            np.testing.assert_allclose(b['models']['I']['estimator'][0].mean_,train_x.mean(axis=0),atol=1e-12)
            for m in b['models'].values():
                for invalid in [np.ones((3200,3)),np.full((3200,3),np.nan),np.full((3200,3),np.inf),np.ones((3,2))]:
                    self.assertEqual(predict_signal(invalid,m,config)[:2],(None,None))
            for group in b['test_groups']:
                selected=self.meta[self.meta.group_id==group].groupby("phase").nth(group%10)
                for r in selected.itertuples():
                    for scenario in self.c['conditions']:
                        w=make_wave(self.w[r.Index],b['scale'],r,scenario)
                        rs=self.p[(self.p.file_id==r.file_id)&(self.p.window_idx==r.window_idx)&(self.p.scenario==scenario)].set_index("model_id")
                        self.assertEqual(hashlib.sha256(w.tobytes()).hexdigest(),rs.input_sha256.iloc[0])
                        for name,m in b['models'].items():
                            score,pred,reason=predict_signal(w,m,config)
                            self.assertEqual(reason,'');self.assertEqual(pred,rs.loc[name,'prediction'])
                            self.assertAlmostEqual(score,rs.loc[name,'score'],places=11)

    def test_home_rerun(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertFalse(list(app.exception))
        app.radio(key='hi_workspace').set_value('슬래그 유동 판정').run()
        self.assertEqual(app.selectbox(key='hi_condition').value,'clean')
        app.selectbox(key='hi_file').set_value('S-11.csv')
        app.slider(key='hi_index').set_value(2).run()
        for scenario,model in [('clean','I'),('noise','H'),('new_z_spike','I')]:
            app.selectbox(key='hi_condition').set_value(scenario);app.selectbox(key='hi_model').set_value(model).run()
            self.assertFalse(list(app.exception));self.assertFalse(list(app.error))
            tab=app.dataframe[0].value
            s=self.p[(self.p.file_id=='S-11.csv')&(self.p.window_idx==2)&(self.p.scenario==scenario)&(self.p.model_id==model)].iloc[0]
            self.assertEqual(tab.iloc[-1]['모델 점수'],f"{s.score:.4f}")
            self.assertEqual(tab.iloc[-1]['임계값'],f"{s.threshold:.4f}")
            self.assertEqual(tab.iloc[-1]['정답과 비교'],'검출' if s.prediction else '미검출')
