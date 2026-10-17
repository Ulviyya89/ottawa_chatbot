"""Run offline regression tests without an API key or paid requests."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
from fastapi.testclient import TestClient
from openai import APIConnectionError
from backend.app.main import app
from backend.app.rag import generator
from backend.app.rag.vector_store import _normalize_category
from backend import vector_store_sqlite as store


class ChatbotTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        self.assertEqual(self.client.get('/health').json(), {'status': 'ok'})

    def test_invalid_messages(self):
        for value in ['', '   ', 'x' * 10001]:
            self.assertEqual(self.client.post('/chat/chat', json={'message': value}).status_code, 422)

    def test_missing_key(self):
        with patch.object(generator.settings, 'openai_api_key', ''):
            response = self.client.post('/chat/chat', json={'message': 'Where can I find housing?'})
        self.assertEqual(response.status_code, 503)

    def test_emergency_path_does_not_call_api(self):
        with patch.object(generator, 'query') as query:
            response = self.client.post('/chat/chat', json={'message': 'This is an emergency'})
        self.assertEqual(response.status_code, 200)
        query.assert_not_called()

    def test_missing_documents(self):
        with patch.object(generator.settings, 'openai_api_key', 'test-key'), patch.object(generator, 'query', return_value=[]):
            response = self.client.post('/chat/chat', json={'message': 'Housing help'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('ingest_local_sqlite.py', response.json()['answer'])

    def test_generated_answer_and_sources(self):
        result = {'text': 'Example housing document.', 'source_file': 'housing.pdf', 'url': '', 'category': 'housing'}
        client = AsyncMock()
        client.responses.create.return_value = SimpleNamespace(output_text='Example housing answer [1].')
        context = AsyncMock()
        context.__aenter__.return_value = client
        with patch.object(generator.settings, 'openai_api_key', 'test-key'), patch.object(generator, 'query', return_value=[result]), patch.object(generator, 'AsyncOpenAI', return_value=context):
            response = self.client.post('/chat/chat', json={'message': 'Housing help', 'category': 'housing'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['sources'][0]['file_name'], 'housing.pdf')
        self.assertIn('[1]', response.json()['answer'])
        self.assertIn('Example housing document.', client.responses.create.call_args.kwargs['input'])

    def test_api_failure(self):
        with patch.object(generator.settings, 'openai_api_key', 'test-key'), patch.object(generator, 'query', side_effect=APIConnectionError(request=httpx.Request('POST', 'https://api.openai.com'))):
            response = self.client.post('/chat/chat', json={'message': 'Housing help'})
        self.assertEqual(response.status_code, 502)

    def test_feedback_matches_interface(self):
        for rating in ['up', 'down']:
            response = self.client.post('/feedback', json={'message': 'Question', 'answer': 'Answer', 'rating': rating, 'timestamp': 'test'})
            self.assertEqual(response.status_code, 200)

    def test_category_mapping(self):
        self.assertEqual(_normalize_category('Community Events'), 'community_events')
        self.assertIsNone(_normalize_category('All'))

    def test_sqlite_retrieval_and_filter(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(store, 'DB_PATH', str(Path(directory) / 'test.sqlite')):
            store.reset_db()
            self.assertEqual(store.query_similar([1., 0.]), [])
            store.add_chunks([
                {'text': 'Housing', 'embedding': [1., 0.], 'source_file': 'housing.pdf', 'category': 'housing'},
                {'text': 'Jobs', 'embedding': [0., 1.], 'source_file': 'jobs.pdf', 'category': 'jobs'},
            ])
            self.assertEqual(store.query_similar([1., 0.], top_k=1)[0]['text'], 'Housing')
            self.assertEqual(store.query_similar([1., 0.], category='jobs')[0]['text'], 'Jobs')

    def test_evaluation_questions(self):
        questions = json.loads((ROOT / 'backend/app/evaluation/test_questions.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(len(questions), 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
