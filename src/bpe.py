# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """    
        # 특수 문자 사전에 등록
        for token, id in SPECIAL_IDS.items():
            # byte_val = val.to_bytes(3, "little")
            self.token_to_id[token] = id
            self.id_to_token[id] = token
        
        # 1바이트 문자 사전에 등록
        for id, token in enumerate(range(0, NUM_BYTES)):
            byte_val = token.to_bytes(1, "little")
            self.id_to_token[id + BYTE_OFFSET] = byte_val
            self.token_to_id[byte_val] = id + BYTE_OFFSET

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        # JSON은 tuple, bytes를 지원하지 않기 때문에 다른 타입으로 저장해야 한다.
        def serialize_token(token):
            if isinstance(token, str):
                return {"type": "str", "value": token}
            if isinstance(token, bytes):
                return {"type": "bytes", "value": list(token)} # b"A" 는 [65] 로 저장
            if isinstance(token, tuple):
                return {"type": "tuple", "value": list(token)}
            raise TypeError(f"JSON으로 저장할 수 없는 token 타입입니다: {type(token)}")

        # token_to_id는 load 쪽에서 만드는것으로 결정
        payload = {
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
            "id_to_token": [],
        }

        for token_id, token in sorted(self.id_to_token.items()):
            payload["id_to_token"].append({
                "id": token_id,
                **serialize_token(token),
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        def deserialize_token(item):
            token_type = item["type"]
            value = item["value"]

            if token_type == "str":
                return value
            if token_type == "bytes":
                return bytes(value)
            if token_type == "tuple":
                return tuple(value)
            raise ValueError(f"알 수 없는 token 타입입니다: {token_type}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.vocab_size = payload["vocab_size"]
        self.merges = [tuple(pair) for pair in payload["merges"]]
        self.id_to_token = {}
        self.token_to_id = {}

        for item in payload["id_to_token"]:
            token_id = item["id"]
            token = deserialize_token(item)
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        # 토큰 ID로 변환
        token_ids = [byte + BYTE_OFFSET for byte in text.encode("utf-8")]

        for pair in self.merges:
            if pair not in self.token_to_id:
                continue

            merged_id = self.token_to_id[pair]
            merged_token_ids = []
            idx = 0

            while idx < len(token_ids):
                if idx < len(token_ids) - 1 and (token_ids[idx], token_ids[idx + 1]) == pair:
                    merged_token_ids.append(merged_id)
                    idx += 2
                else:
                    merged_token_ids.append(token_ids[idx])
                    idx += 1

            token_ids = merged_token_ids

        if add_bos_eos:
            token_ids = [self.get_bos_id()] + token_ids + [self.get_eos_id()]

        return token_ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        raise NotImplementedError("BPETokenizer.decode를 구현하세요.")

# 사전 확인용
if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer._init_special_tokens()

    print(tokenizer.id_to_token)
    print(tokenizer.token_to_id)
