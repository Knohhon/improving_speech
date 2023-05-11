from typing import Iterable
from io import BytesIO
import queue, json, uuid
import multiprocessing.synchronize

from flask import Flask, flash, request, redirect, url_for, make_response


import numpy as np
import vosk
import soundfile as sf
import sentence_transformers
import torch


class RecognizedWord:
    def __init__(self, word: str, confidence: float, start: float, end: float):
        self.word = str(word)
        self.confidence = float(confidence)
        self.start = float(start)
        self.end = float(end)

    @classmethod
    def from_vosk(cls, data: dict):
        return cls(data['word'], data['conf'], data['start'], data['end'])

    @classmethod
    def from_json(cls, json):
        return cls(json[0], json[1], json[2], json[3])

    def to_json(self):
        return self.word, self.confidence, self.start, self.end


class TranscriptResult:
    RESULT_TYPE = 'transcript'

    def __init__(self, words: Iterable[RecognizedWord], full_text: str = None):
        self.words = tuple(words)
        self.full_text = full_text

    @classmethod
    def from_vosk(cls, data: dict):
        words = (RecognizedWord.from_vosk(w) for w in data['result'])
        return cls(words, data['text'])

    @classmethod
    def from_json(cls, json):
        words = tuple(RecognizedWord.from_json(w) for w in json)
        full_text = ' '.join(w.word for w in words)
        return cls(words, full_text)

    def to_json(self):
        return [w.to_json() for w in self.words]


class SimilarityResult:
    RESULT_TYPE = 'similarity'
    def __init__(self, whole_text_similarity: float):
        self.whole_text = float(whole_text_similarity)

    def to_json(self):
        return {'whole' : self.whole_text}


class ResultList:
    def __init__(self, results: Iterable):
        res_map = {}
        for result in results:
            str_type = getattr(result, 'RESULT_TYPE', None)
            if str_type is None:
                raise TypeError('Invalid result type: '+type(result).__name__)

            if str_type in res_map:
                raise KeyError('Multiple results with same type: '+str(result)[:10]+' and '+str(res_map[str_type])[:10])
            res_map[str_type] = result

        self.results = tuple(res_map.values())

    def to_json(self):
        return {result.RESULT_TYPE:result.to_json() for result in self.results}


class AnalyzeResponse:
    def __init__(self, req_id: str = None, error: str = None):
        self.req_id = req_id
        self.error = error

    def to_json(self):
        if self.req_id is None:
            return {'success' : False, 'error' : self.error}
        else:
            return {'success' : True, 'id' : self.req_id}

class ResultResponse:
    def __init__(self, data: ResultList = None, error: str = None):
        self.data = data
        self.error = error

    def to_json(self):
        if self.data is None:
            return {'success' : False, 'error' : self.error}
        else:
            return {'success' : True, 'data' : self.data.to_json()}



class AnalyzeWorker:
    def __init__(self, sample_rate=16000):
        model = vosk.Model(model_name='vosk-model-small-ru-0.22')
        self._sample_rate = int(sample_rate)
        self._rec = vosk.KaldiRecognizer(model, self._sample_rate)
        self._rec.SetPartialWords(True)
        self._rec.SetWords(True)

        self._lm = sentence_transformers.SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cuda')
        self._lm_chunk_len = 256 #self._lm.max_seq_length
        self._lm.max_seq_length = 4096
        self._lm_overlap_size = 16

    def process_long_text_with_overlaps(self, text, output_tokens=True):
        features = self._lm.tokenize([text])

        seq_len = features['input_ids'].shape[1]
        # SequenceLength = Overlap + (ChunkLength - Overlap) * NumChunks
        # We want to find lowest NumChunks, such that the real sequence length is still smaller
        # (basically, we don't want to truncate the text)
        overlap_sz = self._lm_overlap_size
        step_size = self._lm_chunk_len - overlap_sz
        num_chunks = max(1, (seq_len - overlap_sz + step_size - 1) // step_size)
        padded_seq_len = overlap_sz + step_size * num_chunks

        for k in features.keys():
            x = features[k]

            y = x.squeeze(0).to(self._lm.device)
            y = torch.nn.functional.pad(y, (0, padded_seq_len-seq_len), mode='constant')
            y = y.unfold(
                    dimension=0,
                    size=self._lm_chunk_len,
                    step=step_size
                )

            features[k] = y

        self._lm.eval()
        with torch.no_grad():
            out = self._lm(features)
            if output_tokens:
                embeddings = out['token_embeddings']
                del out
                # shape (chunk_index, sequence_index, embedding_dim)

                # Normalize every embedding vector
                embeddings /= torch.norm(embeddings, p=2, dim=-1, keepdim=True)

                # Combine all overlapped chunks by averaging the overlap(could apply a window function here?)
                merged = torch.zeros((padded_seq_len, embeddings.shape[-1]))
                # Manually add the first chunk
                merged[:self._lm_chunk_len, :] = embeddings[0, :, :]
                for chunk_idx in range(1, embeddings.shape[0]):
                    pos = step_size * chunk_idx
                    overlap = embeddings[chunk_idx, :overlap_sz, :]
                    merged[pos:pos + overlap_sz, :] = 0.5 * (
                        merged[pos:pos + overlap_sz, :] + overlap
                    )
                    merged[pos + overlap_sz:pos + self._lm_chunk_len, :] = \
                        embeddings[chunk_idx, overlap_sz:, :]

                # Finally, remove the token embeddings that come from padding and put it back onto the CPU
                final_embeddings = merged[:seq_len].cpu().numpy()
            else:
                embeddings = out['sentence_embedding']
                del out
                # Normalize every embedding vector
                embeddings /= torch.norm(embeddings, p=2, dim=-1, keepdim=True)

                final_embeddings = embeddings.mean(dim=0).cpu().numpy()

        # Normalize once more, since averaging can change length
        final_embeddings /= np.linalg.norm(final_embeddings, ord=2)

        return final_embeddings

    def process(self, audio_file: bytes, target_text: str):
        transcript = next(self.transcribe(audio_file))
        yield transcript

        for res in self.analyze(transcript, target_text):
            yield res

    def transcribe(self, audio_file: bytes):
        with BytesIO(audio_file) as f:
            audio, sample_rate = sf.read(f, dtype='int16', always_2d=True)
            assert sample_rate == self._sample_rate

        audio_data = audio.mean(axis=1, dtype=np.int16).tobytes()

        self._rec.Reset()
        self._rec.AcceptWaveform(audio_data)
        res = json.loads(self._rec.Result())

        yield TranscriptResult.from_vosk(res)

    def analyze(self, transcript: TranscriptResult, target_text: str):
        # embeddings_text = self.process_long_text_with_overlaps(text_file)
        # embeddings_audio = self.process_long_text_with_overlaps(transcript.full_text)

        # embeddings_audio /= np.linalg.norm(embeddings_audio, ord=2, axis=1, keepdims=True)
        # embeddings_text /= np.linalg.norm(embeddings_text, ord=2, axis=1, keepdims=True)

        embedding_text = self.process_long_text_with_overlaps(target_text, output_tokens=False)
        embedding_audio = self.process_long_text_with_overlaps(transcript.full_text, output_tokens=False)

        similarity = (embedding_text * embedding_audio).sum()
        yield SimilarityResult(similarity)


def worker_loop(in_queue: multiprocessing.Queue, out_queue: multiprocessing.Queue, stop: multiprocessing.synchronize.Event):
    w = AnalyzeWorker()
    while not stop.is_set():
        try:
            work_id, work_data = in_queue.get(block=True, timeout=1.0)
        except queue.Empty:
            continue

        for result in work_data:
            out_queue.put((work_id, result))


w = AnalyzeWorker()
app = Flask(__name__)

results = {}
def queue_work(result_generator: Iterable, request_id: str):
    # TODO: Worker threads/processes
    a = list(result_generator)
    results[request_id] = ResultList(a)

@app.route("/process", methods=["POST"])
def process():
    if 'audio' not in request.files:
        return make_response(AnalyzeResponse(error='В запросе нет файла аудио').to_json())

    if 'text' not in request.files:
        return make_response(AnalyzeResponse(error='В запросе нет файла текста').to_json())

    audio = request.files['audio']
    text = request.files['text']

    audio_data = audio.stream.read()
    text_data = text.stream.read().decode('utf-8')

    req_id = str(uuid.uuid4())
    queue_work(w.process(audio_data, text_data), req_id)
    return make_response(AnalyzeResponse(req_id).to_json())

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'audio' not in request.files:
        return make_response(AnalyzeResponse(error='В запросе нет файла аудио').to_json())

    audio = request.files['audio']

    audio_data = audio.stream.read()

    req_id = str(uuid.uuid4())
    queue_work(w.transcribe(audio_data), req_id)
    return make_response(AnalyzeResponse(req_id).to_json())

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'text' not in request.files:
        return make_response(AnalyzeResponse(error='В запросе нет файла текста').to_json())

    text_data = request.files['text']
    transcript_data = request.files['transcript']

    text = text_data.stream.read().decode('utf-8')
    transcript = TranscriptResult.from_json(json.loads(transcript_data.stream.read().decode('utf-8')))

    req_id = str(uuid.uuid4())
    queue_work(w.analyze(transcript, text), req_id)
    return make_response(AnalyzeResponse(req_id).to_json())

@app.route("/result", methods=["GET"])
def get_result():
    req_id = request.args.get('id', default=None)
    if req_id is None:
        return make_response(ResultResponse(error='В запросе нет ID').to_json())

    if req_id not in results:
        return make_response(ResultResponse(error='Несуществующий ID запроса').to_json())

    result_list = results.pop(req_id)
    return make_response(ResultResponse(result_list).to_json())


if __name__ == "__main__":
    app.run()
