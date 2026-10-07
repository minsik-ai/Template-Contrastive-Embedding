import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import publish_private_dataset as publisher


class PrivatePublishTests(unittest.TestCase):
    def fixture(self, root):
        manifest = root / "generation.json"
        manifest.write_text("{}")
        report = {"eligible_for_private_upload": True, "packaged": True, "failures": [],
                  "manifest": {"path": str(manifest), "sha256": publisher.gate.sha256(manifest)},
                  "parquet_files": []}
        (root / "data").mkdir()
        for name in publisher.gate.PAPER:
            path = root / "data" / f"{name}-train.parquet"
            path.write_bytes(b"test package payload")
            report["parquet_files"].append({"path": str(path.relative_to(root)),
                                            "sha256": publisher.gate.sha256(path)})
        (root / "README.md").write_text("Test card")
        (root / "verification.json").write_text(json.dumps(report))
        return report

    def test_failed_gate_never_calls_hub(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "verification.json").write_text(json.dumps({"eligible_for_private_upload": False}))
            api = Mock()
            with self.assertRaisesRegex(ValueError, "verification and packaging"):
                publisher.publish(root, "owner/test", api)
            self.assertEqual(api.mock_calls, [])

    def test_changed_package_and_failed_fresh_audit_never_call_hub(self):
        for fresh_ok in (False, True):
            with self.subTest(fresh_ok=fresh_ok), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root)
                (root / "data/snips-train.parquet").write_bytes(b"modified")
                api = Mock()
                with (
                    patch.object(publisher.gate, "audit", return_value={"eligible_for_private_upload": fresh_ok}),
                    self.assertRaises(ValueError),
                ):
                    publisher.publish(root, "owner/test", api)
                self.assertEqual(api.mock_calls, [])

    def test_private_creation_required_and_unrelated_files_not_uploaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / "unrelated-secret.txt").write_text("never upload")
            api = Mock()
            api.whoami.return_value = {"name": "owner"}
            api.dataset_info.return_value = SimpleNamespace(private=True)
            api.upload_folder.return_value = SimpleNamespace(oid="abc123")
            api.list_repo_files.return_value = ["README.md", "verification.json"] + [
                f"data/{name}-train.parquet" for name in publisher.gate.PAPER]
            with patch.object(publisher.gate, "audit", return_value={"eligible_for_private_upload": True}):
                receipt = publisher.publish(root, "owner/test", api)
            api.create_repo.assert_called_once_with(repo_id="owner/test", repo_type="dataset",
                                                   private=True, exist_ok=False)
            self.assertNotIn("unrelated-secret.txt", api.upload_folder.call_args.kwargs["allow_patterns"])
            self.assertTrue(receipt["private"])

    def test_public_destination_stops_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            api = Mock()
            api.whoami.return_value = {"name": "owner"}
            api.dataset_info.return_value = SimpleNamespace(private=False)
            with (
                patch.object(publisher.gate, "audit", return_value={"eligible_for_private_upload": True}),
                self.assertRaisesRegex(RuntimeError, "not confirmed private"),
            ):
                publisher.publish(root, "owner/test", api)
            api.upload_folder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
