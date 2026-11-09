'''Verify recovery, evidence routing and company isolation with local tool doubles.'''

import asyncio
import unittest
from unittest.mock import patch

import hw3


def assessment(verdict='PASS', uncertain=False, override=None):
    rows = []
    for point, route in hw3.POINT_ROUTES.items():
        rows.append(dict(point=point, verdict=(override or {}).get(point, verdict),
                         source_ids=[f'p{point}_{route[1]}'], uncertain=uncertain,
                         explanation='the knowledgeable witness describes concrete practices that support the selected rule and provide relevant evidence for this company'))
    return hw3.CompanyAssessment.model_validate({'assessments': rows})


class Interview:
    def __init__(self, unavailable=False, failure=False):
        self.calls = []
        self.unavailable = unavailable
        self.failure = failure

    async def ainvoke(self, arguments):
        self.calls.append(arguments)
        if self.failure:
            raise ConnectionError('Tool temporarily unavailable')
        if self.unavailable and arguments['role'] == 'current_employee':
            return 'I cannot comment on this topic.'
        return 'The witness describes concrete and relevant business practices.'


class Classifier:
    def __init__(self, answers):
        self.answers = list(answers)
        self.messages = []

    async def ainvoke(self, messages):
        self.messages.append(messages)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return {'parsed': answer, 'raw': None}


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, classifier, interview=None, reflection=True):
        tool = interview or Interview()
        agent = hw3.FisherAgent(classifier=classifier, reflection=reflection)
        result = await hw3.analyze_companies(agent, {'unseen_company': [None, None, None, tool]})
        return result['unseen_company'], agent, tool

    async def test_routing_and_independent_review(self):
        classifier = Classifier([assessment(), assessment()])
        result, agent, tool = await self.run_case(classifier)
        self.assertEqual(result['points'], {4: 'PASS', 7: 'PASS', 11: 'PASS'})
        self.assertEqual(tool.calls, [dict(role=route[1], topic=route[0]) for route in hw3.POINT_ROUTES.values()])
        self.assertEqual(len(classifier.messages), 2)
        self.assertNotIn('initial', classifier.messages[1][1][1])
        self.assertEqual(agent.traces['unseen_company']['status'], 'complete')

    async def test_disagreement_triggers_relevant_backup(self):
        classifier = Classifier([assessment(), assessment(override={4: 'FAIL'}),
                                 assessment(override={4: 'FAIL'})])
        result, agent, tool = await self.run_case(classifier)
        self.assertEqual(result['points'][4], 'FAIL')
        self.assertEqual(len(tool.calls), 4)
        self.assertEqual(tool.calls[-1], dict(role='competitor', topic='sales_team'))
        self.assertEqual(agent.traces['unseen_company']['disputed_points'], [4])

    async def test_unavailable_witness_gets_knowledgeable_backup(self):
        first = assessment()
        first.assessments[1].source_ids = ['p7_former_employee']
        result, agent, tool = await self.run_case(Classifier([first]), Interview(unavailable=True), False)
        self.assertEqual(result['points'][7], 'PASS')
        self.assertEqual(len(tool.calls), 4)
        self.assertEqual(tool.calls[-1], dict(role='former_employee', topic='working_conditions'))

    async def test_review_failure_retains_initial_verdicts(self):
        classifier = Classifier([assessment(), RuntimeError(), RuntimeError()])
        result, agent, _ = await self.run_case(classifier)
        self.assertEqual(result['points'][4], 'PASS')
        self.assertEqual(agent.traces['unseen_company']['status'], 'degraded')

    async def test_failed_company_keeps_valid_output(self):
        result, agent, _ = await self.run_case(Classifier([RuntimeError(), RuntimeError()]))
        self.assertEqual(result['points'], {4: 'FAIL', 7: 'FAIL', 11: 'FAIL'})
        self.assertIn('operational failure', result['summary'])
        self.assertEqual(agent.traces['unseen_company']['status'], 'failed')

    async def test_timeout_isolated_between_companies(self):
        agent = hw3.FisherAgent(classifier=None, concurrency=2, company_timeout=0.02)
        async def work(agent, tools, trace):
            if tools[0] == 'slow':
                await asyncio.sleep(0.1)
            return hw3._as_result(assessment())
        with patch.object(hw3, '_company_analysis', work):
            result = await hw3.analyze_companies(agent, {'slow': ['slow'], 'fast': ['fast']})
        self.assertEqual(result['fast']['points'][4], 'PASS')
        self.assertEqual(agent.traces['slow']['status'], 'failed')
        self.assertEqual(agent.traces['fast']['status'], 'complete')

    async def test_invalid_cross_point_reference_retries(self):
        wrong = assessment()
        wrong.assessments[0].source_ids = ['p7_current_employee']
        classifier = Classifier([wrong, assessment()])
        result, agent, _ = await self.run_case(classifier, reflection=False)
        self.assertEqual(result['points'][4], 'PASS')
        self.assertEqual(agent.traces['unseen_company']['calls'][0]['error'], 'ValueError')

    async def test_duplicate_points_rejected(self):
        data = assessment().model_dump()
        data['assessments'][2]['point'] = 4
        with self.assertRaises(ValueError):
            hw3.CompanyAssessment.model_validate(data)

    async def test_summary_contrast_removed(self):
        text = 'The witness reports heavy workload rather than healthy personnel relations.'
        self.assertEqual(hw3._prose(text), 'the witness reports heavy workload')
        text = 'The sales activity is not technical advice but basic order taking.'
        self.assertEqual(hw3._prose(text), 'the sales activity is basic order taking')

    async def test_summary_punctuation(self):
        item = assessment()
        item.assessments[0].explanation = 'The witness describes fair practices. These practices support the rule.'
        result = hw3._as_result(item)
        self.assertEqual(result['summary'].count('.'), 3)
        self.assertNotIn(chr(34), result['summary'])
        self.assertNotIn(chr(0x2014), result['summary'])


if __name__ == '__main__':
    unittest.main()
