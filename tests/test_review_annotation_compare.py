"""Read-only annotation UI and data projection contracts; no model calls."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts_omnidance/exp/review_web/build_annotation_compare.py"
spec = importlib.util.spec_from_file_location("annotation_builder", SOURCE)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class AnnotationCompareTests(unittest.TestCase):
    def test_MS_original_text_and_empty_abstain_preserved(self):
        text = "<script>Not HTML</script> repeated movement."
        view = dict(window_start=0., window_end=2., status="candidate", abstain_reason=None,
                    cleaned_motion_text=text, partial_supported_motion_text="",
                    original_captions=[dict(start=0., end=1.95, ms=[text], ms_nonagg=["Original wording."])])
        projected = builder.ms_view("BPM", [view])["windows"][0]
        self.assertEqual(projected["caption"], text)
        self.assertEqual(projected["raw_original_MS"][0]["ms_nonagg"], ["Original wording."])
        view.update(status="abstain", abstain_reason="no_output", cleaned_motion_text="")
        self.assertEqual(builder.ms_view("BPM", [view])["windows"][0]["caption"], "")

    def test_new_points_and_body_layout(self):
        self.assertEqual(len(builder.BODY_PARENTS), 22)
        covered = {i for edge in builder.EXTRA_EDGES for i in edge if i >= 22}
        self.assertEqual(covered, set(range(22, 32)))
        self.assertFalse(set(builder.SKELETON_IDS) & set(builder.CHUNK_IDS))

    def test_new_UI_uses_text_nodes_and_does_not_write_reviews(self):
        html = (ROOT / "scripts_omnidance/exp/review_web/index.html").read_text()
        code = html.split("/* Independent, read-only annotation view:", 1)[1]
        self.assertIn("textContent=String(text)", code)
        self.assertNotIn("innerHTML", code)
        self.assertNotIn("method:\"POST\"", code)
        self.assertNotIn("localStorage", code)
        self.assertIn("sourceFilter", code)
        self.assertIn("#annotation", code)
        self.assertIn("view.windows.forEach", code)


if __name__ == "__main__":
    unittest.main()
