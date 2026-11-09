"""Local presentation only. No model inputs are created in this module."""
from base64 import b64encode
from html import escape
from pathlib import Path

import streamlit as st

from .surface_replay import png

MODELS = {"V0":"기존 모델 V0", "G1":"개선 모델 G1"}
CHANGES_KO = {"fixed":"바로잡은 오류", "new_error":"새로 발생한 오류",
              "both_wrong":"두 모델 모두 틀린 사례", "both_correct":"두 모델 모두 맞힌 사례"}
MESSAGES = {"fixed":"개선 모델이 기존 오류를 바로잡았습니다.",
            "new_error":"개선 모델에서 새 오류가 발생했습니다.",
            "both_wrong":"두 모델 모두 이 입력을 잘못 분류했습니다.",
            "both_correct":"두 모델 모두 이 입력을 올바르게 분류했습니다."}
STRENGTH = {0.0:"원본", 0.02:"약한 잡음", 0.05:"중간 잡음", 0.10:"강한 잡음"}


def apply_style():
    st.html(Path(__file__).with_name("demo.css"))


def brand():
    st.html('<div class="steel-brand"><span class="steel-symbol" aria-hidden="true"><i></i><i></i><i></i></span>'
            '<div><b>SteelTrust</b><small>강판 표면 결함 검증</small></div></div>')


def image_uri(raw):
    return "data:image/png;base64," + b64encode(png(raw)).decode("ascii")


def heading(title, description=None):
    st.html(f'<header class="steel-heading"><h1>{escape(title)}</h1>'
            + (f'<p>{escape(description)}</p>' if description else "") + '</header>')


def outcome_title(label, before, after, change):
    def particle(word, consonant, vowel):
        tail = (ord(word[-1]) - 0xAC00) % 28
        return word + (consonant if tail and not (consonant == "으로" and tail == 8) else vowel)
    actual = particle(label,"을","를")
    if change == "fixed":
        return f"기존 모델이 {actual} {particle(before,'으로','로')} 잘못 분류한 사례"
    if change == "new_error":
        return f"개선 모델이 {actual} {particle(after,'으로','로')} 잘못 분류한 사례"
    if change == "both_wrong":
        return f"두 모델 모두 {actual} 잘못 분류한 사례"
    return f"두 모델 모두 {actual} 올바르게 분류한 사례"


def prediction_card(model, label, correct):
    verdict = "정답" if correct else "오분류"
    tone = "right" if correct else "wrong"
    return (f'<div class="steel-pred {tone}"><span>{MODELS[model]}</span><strong>{escape(label)}</strong>'
            f'<em>{verdict}</em></div>')
