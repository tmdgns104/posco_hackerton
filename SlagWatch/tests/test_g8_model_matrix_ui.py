"""Streamlit-free smoke tests of all tabs, backed by real saved predictions.

Does not replace a full Windows/Streamlit browser test.
"""
import importlib
import sys
import types
import unittest
from unittest.mock import patch

class NoOpCtx:
    def __enter__(self):return self
    def __exit__(self,*_):return False
    def metric(self,*a,**kw):pass

class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__('streamlit')
        self.choice={}
        self.events=[]
    def __getattr__(self,attr):
        if attr in ('tabs',):return lambda values:[NoOpCtx() for _ in values]
        if attr=='columns':return lambda values:[NoOpCtx() for _ in range(values if isinstance(values,int) else len(values))]
        if attr in ('expander','container'):return lambda *a,**kw:NoOpCtx()
        if attr in ('radio','selectbox'):
            def choose(label,options,*args,**kwargs):
                opts=list(options)
                key=kwargs.get('key',label)
                value=self.choice.get(key,opts[kwargs.get('index',0)])
                assert value in opts,(key,value,opts)
                return value
            return choose
        if attr=='file_uploader':return lambda *a,**kw:None
        if attr=='checkbox':return lambda *a,**kw:False
        if attr=='button':return lambda *a,**kw:False
        if attr in ('bar_chart','line_chart','dataframe'):
            def check(data,*a,**kw):
                assert data is not None
                self.events.append((attr,len(data)))
            return check
        if attr in ('image','markdown','caption','metric','info','warning','success','error','code','write','subheader','download_button'):
            return lambda *a,**kw:None
        raise AttributeError(attr)

class TestG82UI(unittest.TestCase):
    def setUp(self):
        self.fake=FakeStreamlit()
        self.patch=patch.dict(sys.modules,{'streamlit':self.fake})
        self.patch.start()
        from src import g8_model_matrix_ui
        self.ui=g8_model_matrix_ui
        self.ui.st=self.fake
    def tearDown(self):self.patch.stop()

    def test_surface_four_tabs_and_pair(self):
        self.ui.surface_screen()
        self.assertGreaterEqual(len(self.fake.events),5)

    def test_surface_v0_g1_test_scope(self):
        for mode in ('overview','single','paired'):
            self.fake.choice[f'g8_surface_dataset_{mode}']='V0·G1 최종 시험 (Test)'
        self.ui.surface_screen()
        self.assertGreaterEqual(len(self.fake.events),5)

    def test_binary_tabs(self):
        self.fake.choice['검사 종류']='정상·불량 검사 · K0/K1'
        self.ui.surface_screen()
        self.assertGreaterEqual(len(self.fake.events),2)

    def test_slag_eight_and_paired(self):
        self.ui.slag_screen()
        self.assertGreaterEqual(len(self.fake.events),6)

    def test_slag_g3r_individual(self):
        self.fake.choice['검사할 모델']='G3R'
        self.ui.slag_screen()
        self.assertGreaterEqual(len(self.fake.events),6)

if __name__=='__main__':unittest.main()
