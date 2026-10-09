import qdrant_client.http.exceptions
from hstest import StageTest, TestedProgram, CheckResult, dynamic_test
import re
import requests
import json

class MovieQAToolTest(StageTest):
    test_data = [
        (
            "Mission Impossible",
            129,
            r"Loaded script for Mission Impossible from https://imsdb\.com/scripts/Mission-Impossible\.html\.",
            "Create a scene depicting Ethan as the antagonist and living in a cave. It should start with INT. CAVE - DAY",
            r"CAVE|ETHAN|ANTAGONIST|WATER|INT"
        )
    ]

    @staticmethod
    def check_qdrant_connection():
        """Checks connectivity with the Qdrant server."""
        try:
            response = requests.get("http://localhost:6333/collections")
            if response.status_code != 200:
                return "Cannot connect to the Qdrant server. Make sure it is running."
        except requests.RequestException:
            return "Cannot connect to the Qdrant server. Make sure it is running."
        return None

    @staticmethod
    def verify_qdrant_collection(collection):
        """Validates the Qdrant collection for the given movie."""
        try:
            response = requests.get(f"http://localhost:6333/collections/{collection}")
            data = response.json()
        except qdrant_client.http.exceptions.UnexpectedResponse as e:
            return f"Unable to create a collection for the script of {collection.replace('-', ' ')}. Encountered: {e}"
        except qdrant_client.http.exceptions.ResponseHandlingException as e:
            return f"Cannot connect to the Qdrant server. Make sure it is running. Encountered: {e}"
        except json.JSONDecodeError:
            return "Response is not valid JSON."
        except Exception as e:
            return f"An error occurred while loading the collection for {collection.replace('-', ' ')}. Encountered: {e}!"

        expected_keys = {'status', 'time', 'result'}
        if not expected_keys.issubset(data.keys()):
            return f"Response JSON does not contain the expected keys: {expected_keys}"
        if data.get('status') != 'ok':
            return f"Expected status 'ok', but got '{data.get('status')}'"

        result = data.get('result', {})
        config = result.get('config')
        if not isinstance(config, dict):
            return "'config' is not a dictionary in 'result'"
        params = config.get('params')
        if not isinstance(params, dict):
            return "'params' is missing or not a dictionary in 'config'"
        vectors = params.get('vectors')
        if not isinstance(vectors, dict):
            return "'vectors' is missing or not a dictionary in 'params'"

        if vectors.get('distance') != 'Cosine':
            return f"Expected distance to be 'Cosine', but got '{vectors.get('distance')}'"
        if vectors.get('size') not in [384, 1536]:
            return f"Expected size to be 384 or 1536, but got '{vectors.get('size')}'"
        if result.get('points_count', 0) == 0:
            return f"The collection '{collection}' does not contain anything"
        return None

    @dynamic_test(time_limit=0)
    def test1_loading_and_generate_response(self):
        for movie, scene_count, expected_info_regex, query, expected_keywords_regex in self.test_data:
            program = TestedProgram("main")
            # Simulate entering the movie title.
            program.start()
            output = program.execute(movie)

            # Check for the expected info line (movie title and script URL).
            if not re.search(expected_info_regex, output):
                return CheckResult.wrong(
                    f"For movie '{movie}', the expected info about the script URL was not found.\n"
                )

            # Check for scene count information.
            expected_scene_info = f"Found {scene_count} scenes in the script for {movie}."
            if expected_scene_info not in output:
                return CheckResult.wrong(
                    f"For movie '{movie}', expected scene count info '{expected_scene_info}' was not found.\n"
                )

            # Check that the embedded script info is present.
            embedded_info = f"Embedded script for {movie}."
            if embedded_info not in output:
                return CheckResult.wrong(
                    f"For movie '{movie}', expected embedded script info '{embedded_info}' was not found.\n"
                )

            # Simulate entering the fixed query.
            output2 = program.execute(query)

            # Check that the final answer section is present.
            if "Final Answer:" not in output2:
                return CheckResult.wrong(
                    f"For movie '{movie}', the final answer section ('Final Answer:') was not found in the output.\n"
                )

            # Verify that the final answer contains the expected keywords.
            if not re.search(expected_keywords_regex, output2, re.IGNORECASE):
                return CheckResult.wrong(
                    f"For movie '{movie}', the final answer does not contain the expected keywords.\n"
                    f"Expected pattern: {expected_keywords_regex}\n"
                )
        return CheckResult.correct()


if __name__ == '__main__':
    MovieQAToolTest().run_tests()