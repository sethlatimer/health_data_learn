import json
import unittest
from unittest.mock import Mock, patch

from agent import Agent


class AgentIntentRouterTests(unittest.TestCase):
    def test_examples_select_expected_routes_and_return_responses(self):
        examples = (
            (
                "document_search",
                "Summarize the attached PDF for me.",
                "Document summary response",
            ),
            (
                "document_search",
                "What does the report say about patient safety?",
                "Document evidence response",
            ),
            (
                "oecd_discovery",
                "Which OECD datasets cover primary care?",
                "OECD discovery response",
            ),
            (
                "oecd_discovery",
                "Show me available OECD healthcare measures.",
                "OECD measures response",
            ),
            (
                "oecd_analysis",
                "How have asthma admissions changed in Australia since 2016?",
                "OECD analysis response",
            ),
            (
                "oecd_analysis",
                "Did diabetes hospitalizations increase from 2018 to 2023?",
                "Deterministic trend response",
            ),
            (
                "general_chat",
                "Hello!",
                "General greeting response",
            ),
            ("general_chat", "Thank you for your help.", "General thanks response"),
        )

        for expected_route, question, final_response in examples:
            with self.subTest(route=expected_route, question=question):
                agent = Agent()
                response_by_route = {
                    "document_search": "ask_document_corpus",
                    "oecd_discovery": "ask_oecd_discovery",
                    "oecd_analysis": "ask_oecd_analysis",
                    "general_chat": "ask",
                }
                with patch.object(
                    agent,
                    response_by_route[expected_route],
                    return_value=final_response,
                ):
                    routing = agent.route_request(question)
                    response = agent.ask_corpus(question)
                    print(f"Selected route: {routing['route']}")
                    print(f"Final response: {response}")

                self.assertEqual(routing, {"route": expected_route})
                self.assertEqual(response, final_response)

    @patch("agent.requests.post")
    def test_ambiguous_request_uses_validated_json_route(self, post):
        response = Mock()
        response.json.return_value = {"response": '{"route":"general_chat"}'}
        post.return_value = response

        routing = Agent().route_request("What is the best way to learn statistics?")

        self.assertEqual(routing, {"route": "general_chat"})
        payload = post.call_args.kwargs["json"]
        self.assertIn("format", payload)
        self.assertEqual(payload["format"]["required"], ["route"])
        self.assertFalse(payload["format"]["additionalProperties"])

    @patch("agent.requests.post")
    def test_ambiguous_router_rejects_prose_or_extra_fields(self, post):
        response = Mock()
        response.json.return_value = {
            "response": json.dumps({
                "route": "oecd_analysis",
                "dataset_id": "invented",
            })
        }
        post.return_value = response

        with self.assertRaisesRegex(ValueError, "invalid routing object"):
            Agent().route_request("Can you look into this for me?")


if __name__ == "__main__":
    unittest.main(verbosity=2)
