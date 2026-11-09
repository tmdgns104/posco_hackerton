"""Isolated Streamlit route; original sealed research modules remain byte-identical."""
import json

import numpy as np
import pandas as pd
import streamlit as st

from .adaptive_audit import audit
from .adaptive_redteam import (
    LOW, HIGH, image_change, run_surface, seed_of, summary_table,
)
from .surface_data import ROOT, RUN, CFG
from .surface_detector import read_raw


def workbench():
    st.title('Adaptive Red Team · 취약 조건 능동 탐색')
    st.caption('SteelTrust G2 · 기존 AI는 읽기 전용 · 검증(validation) 데이터만 사용 · 독립 시험 결과 아님')
    explorer, signal_explorer, inventory=st.tabs(['강판 결함 AI 능동 탐색','진동 AI 능동 탐색','전체 모델 검증 기록'])
    with inventory:
        info=audit(ROOT)
        if info['n_failed']:
            st.error(f"모델/원본/코드 해시 검사 실패: {info['n_failed']}건. 실행을 중단하고 원본을 확인하세요.")
        else:
            st.success(f"기존 연구 파일 해시 점검 {info['n_checked']}개 통과")
        st.dataframe(pd.DataFrame([
            {'모델':'G1','판정':'기존 검증 통과','설명':'표면 결함 강건성 개선, clean 성능 감소도 함께 기록'},
            {'모델':'V0','판정':'기준 모델','설명':'원본 정확도 우수, 잡음에 취약'},
            {'모델':'V1·V2','판정':'검증 미통과','설명':'일부 조건에서 성능 하락'},
            {'모델':'K0·K1','판정':'K1 미채택','설명':'불량 미탐 증가, 공식 KSDD2 이미지 미첨부'},
            {'모델':'CNN C0·C1','판정':'검증 미통과','설명':'슬래그 진동 감지 성능 부족'},
            {'모델':'기존 artifacts','판정':'비교 기준','설명':'4개 모델 파일과 그룹 매핑 확인'},
        ]),hide_index=True,width='stretch')
        with st.expander('해시 검사 세부 목록'):
            st.dataframe(pd.DataFrame(info['checks']),hide_index=True,width='stretch')
        st.download_button('검증 결과 JSON',json.dumps(info,ensure_ascii=False,indent=2).encode('utf8'),
                           'adaptive_model_audit.json','application/json',on_click='ignore')
        st.warning('기존 K1, V1/V2, CNN 후보를 사용자 승인 없이 자동 채택하지 않습니다. 기존 연구 결과 파일과 가중치는 수정하지 않습니다.')
    with explorer:
        info=audit(ROOT)
        if info['n_failed']:
            st.error('해시 검증에 실패했습니다. 모델을 실행하지 않습니다.');return
        st.info('현재 능동 탐색은 NEU-CLS 6종 분류 모델(V0/V1/V2/G1)용입니다. K0/K1은 원본 KSDD2 이미지·마스크 부재로 탐색이 비활성화됩니다. CNN은 기존 검증 기록을 보존하며 이 화면의 영상 교란 시험에 연결하지 않습니다.')
        meta=pd.read_csv(RUN/'split_manifest.csv')
        val=meta[meta.split=='validation'].sort_values('sample_id')
        model=st.selectbox('검사 대상', ['V0','G1','V1','V2'])
        sample_count=st.slider('시험 이미지 수 (검증 집합)',1,12,4)
        groups=[val[val.label==cl].sample_id.tolist() for cl in sorted(CFG['classes'].values())]
        ids=[g[i] for i in range(2) for g in groups if len(g)>i]
        choices=ids[:sample_count]
        select=st.multiselect('시험 이미지 ID',val.sample_id.tolist(),default=choices,max_selections=12)
        c1,c2,c3=st.columns(3)
        budget=c1.slider('이미지당 탐색 횟수',8,60,20,step=4)
        warmup=c2.slider('동일한 랜덤 준비 시험',4,min(12,budget),min(8,budget))
        seed=c3.number_input('재현 seed',min_value=0,max_value=2_147_483_647,value=2026,step=1)
        compare=st.checkbox('같은 예산의 무작위 탐색과 비교',value=True)
        st.caption('교란: 가우시안 잡음 0~0.10, 밝기 ±0.12, 대비 0.76~1.24, 블러 σ 0~1.2. RMSE와 포화 비율 기준을 넘는 입력은 무효로 기록합니다. 실험별 정답 보존 여부는 별도 육안 확인이 필요합니다.')
        if st.button('실제 모델로 Adaptive Red Team 실행',type='primary'):
            if not select:st.error('검증 이미지 하나 이상을 선택하세요.');return
            progress=st.progress(0,text='모델 검사 준비')
            try:
                output=run_surface(ROOT,select,model_id=model,budget=budget,warmup=warmup,
                                   seed=int(seed),compare_random=compare,
                                   progress=lambda i,total: progress.progress(i/total,text=f'검증 이미지 {i}/{total}'))
            except Exception as e:
                st.error(f'실행 오류: {type(e).__name__}: {e}')
                return
            finally:
                progress.empty()
            st.session_state['adaptive_redteam_result']=output
        output=st.session_state.get('adaptive_redteam_result')
        if output is None:
            st.caption('실행 전에는 과거 G1 연구 결과를 변경하거나 새 모델의 성능을 표시하지 않습니다.')
            return
        df=summary_table(output)
        st.subheader('탐색 결과 · 독립 성능 평가가 아닌 개발 실험')
        st.caption(f"마지막 실행 결과: {output['model_id']} · seed {output['rng_seed']} (현재 입력 위젯을 변경해도 재실행 전까지 그대로 표시)")
        if df.empty:
            st.warning('원본부터 오분류인 이미지 등으로 유효한 탐색 결과가 없습니다.')
        else:
            st.dataframe(df,hide_index=True,width='stretch')
            count=df.groupby('strategy').agg(이미지수=('sample_id','size'),발견한_이미지수=('failures',lambda x:int((x>0).sum())),
                                             총실패시험=('failures','sum')).reset_index()
            st.table(count)
        skipped=[r for r in output['runs'] if 'skipped' in r]
        if skipped:
            st.warning(f'원본이 이미 오분류되어 유발 실패의 분모에서 제외한 이미지 {len(skipped)}장: '+', '.join(r['sample_id'] for r in skipped))
        st.caption('동일한 이미지에 반복한 시험은 독립 표본이 아닙니다. 발견한 실패 횟수를 이미지 정확도로 해석해서는 안 됩니다.')
        st.download_button('결과 전체 JSON',json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False).encode('utf8'),
                           'adaptive_redteam_results.json','application/json',on_click='ignore')
        if not df.empty:
            st.download_button('비교 요약 CSV',df.to_csv(index=False).encode('utf-8-sig'),
                               'adaptive_redteam_summary.csv','text/csv',on_click='ignore')
            rows=[]
            for run in output['runs']:
                for trial in run.get('trials',[]):
                    rows.append({k:v for k,v in dict(sample_id=run['sample_id'],model_id=run['model_id'],strategy=run['strategy'],
                                                     **trial,**trial['parameters']).items()
                                 if k not in ('parameters','parameters_vector','scores')})
            st.download_button('전체 시험 기록 CSV',pd.DataFrame(rows).to_csv(index=False).encode('utf-8-sig'),
                               'adaptive_redteam_trials.csv','text/csv',on_click='ignore')
        discovered=[(run,t) for run in output['runs'] for t in run.get('trials',[]) if t.get('wrong')]
        if discovered:
            run,trial=min(discovered,key=lambda v:v[1]['rmse'])
            st.subheader('발견된 최소 교란 오분류 사례 · 시각적 검토 필요')
            r=val.set_index('sample_id').loc[run['sample_id']]
            raw=read_raw(r)
            noise=np.random.default_rng(seed_of('noise',run['sample_id'],run['settings']['trial_seed'])).normal(size=raw.shape).astype(np.float32)
            altered,_=image_change(raw,trial['parameters_vector'],noise)
            a,b=st.columns(2)
            a.image(raw,clamp=True,caption=f"원본 · 정답 {run['label']}",width='stretch')
            b.image(altered,clamp=True,caption=f"오분류 {trial['prediction']} · RMSE {trial['rmse']:.4f}",width='stretch')
            st.json({'sample_id':run['sample_id'],'model':run['model_id'],'strategy':run['strategy'],
                     'trial':trial['trial'],'parameters':trial['parameters'],'input_sha256':trial['input_sha256']})
        st.warning('본 기능은 연구용 취약성 탐색입니다. 결함 의미를 보존한다는 자동 보증은 없으며, 재학습/모델 교체/현장 제어는 실행하지 않습니다.')
    with signal_explorer:
        st.subheader('슬래그 유동 진동 AI · 능동 교란 탐색')
        st.warning('이 실험은 이전에 평가된 그룹 교차검증 데이터를 다시 사용합니다. 결과는 후속 개발 탐색이며 새로운 독립 시험 성능이 아닙니다.')
        from .adaptive_signal import run_signal
        from .data import recording_paths
        ids=sorted(recording_paths(ROOT/'data/raw'))
        c1,c2,c3=st.columns(3)
        name=c1.selectbox('공식 SSFD 파일', ids, index=ids.index('S-4.csv') if 'S-4.csv' in ids else 0)
        idx=c2.slider('0.5초 구간',0,9,0)
        model_id=c3.selectbox('검사 모델',('C0','C1','A1','J'),index=1)
        t1,t2=st.columns(2)
        budget2=t1.slider('진동 시험 횟수',8,40,16,step=4)
        seed2=t2.number_input('진동 시험 seed',min_value=0,max_value=2_147_483_647,value=42)
        if st.button('진동 모델 Adaptive Red Team 실행'):
            try:
                result=run_signal(file_id=name,window_idx=idx,model_id=model_id,
                                  budget=budget2,warmup=min(8,budget2),seed=int(seed2))
                st.session_state['adaptive_signal_result']=result
            except Exception as err:
                st.error(f'{type(err).__name__}: {err}')
        result=st.session_state.get('adaptive_signal_result')
        if result:
            if result.get('skip'):
                st.warning(f"원본부터 오답이므로 유발 실패 실험에서 제외: {result['file_id']} / {result['model_id']}")
            else:
                st.table(pd.DataFrame([{k:v for k,v in item.items() if k!='trials'} for item in result['results']]))
                rows=[]
                for item in result['results']:
                    for trial in item['trials']:
                        rows.append(dict(strategy=item['strategy'],trial=trial['trial'],
                                         wrong=trial['wrong'],relative_rmse=trial['relative_rmse'],
                                         **trial['parameters']))
                st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
                st.caption(f"교차검증 체크포인트: {result['model_id']} fold {result['fold']} · 입력: {result['file_id']} 구간 {result['window_idx']}")
            st.download_button('진동 탐색 결과 JSON',json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False).encode('utf8'),
                               'adaptive_signal_results.json','application/json',on_click='ignore')
            st.caption('잡음·축별 감도·충격의 합성 주입은 물리적으로 같은 유동 상태라는 보증이 없습니다. 현장 자동 제어에는 사용하지 마세요.')
