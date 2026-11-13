"""Immutable, self-contained LOCAL expert samples shared by baseline modes.

The original complete started/raw journal records remain embedded. Only the
operational derived-audio path is excluded from cross-run request comparison;
waveform bytes, parent range, clock, seed, budget and every other field match.
This is reuse of recorded outputs, not a claim to have rerun the original model.
"""
import copy
from pathlib import Path

from .audio import inspect_audio, make_windows
from .io import file_lock, file_sha, object_sha, read_json_snapshot, write_json
from .schema import validate_dataset


LOCAL_CONFIG = ("seed", "max_output_tokens", "window_seconds", "stride_seconds",
                "candidates_per_window", "local_provider")


def _same(left, right):
    return object_sha(left) == object_sha(right)


def local_settings(config):
    return {key: copy.deepcopy(config[key]) for key in LOCAL_CONFIG}


def audio_identity(audio):
    return {key: audio[key] for key in ("sha256", "sample_rate", "num_samples")}


def _stat(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def canonical_local_request(request):
    required = {"call_id", "seed", "max_output_tokens", "response_kind", "audio"}
    audio_keys = {"path", "sha256", "sample_rate", "num_samples", "parent_audio_sha256",
                  "parent_start", "parent_end", "conversion"}
    if (not isinstance(request, dict) or set(request) != required
            or request["response_kind"] != "local_caption"
            or not isinstance(request["audio"], dict) or set(request["audio"]) != audio_keys
            or not isinstance(request["audio"]["path"], str) or not request["audio"]["path"]):
        raise ValueError("Candidate bank requires the exact local audio-only request contract")
    value = copy.deepcopy(request)
    value["audio"].pop("path")
    return value


def _source_context(contract):
    from .baseline import _config
    if contract.get("schema") != "choreobench.run_contract.v1":
        raise ValueError("Candidate source is not a baseline run contract")
    if contract.get("local_candidates_identity"):
        raise ValueError("Export from the original collection run; do not repackage a reused bank")
    dataset = validate_dataset(contract["dataset_snapshot"])
    cfg = _config(contract["config"])
    songs = [song for song in dataset["songs"] if song["split"] in cfg["splits"]]
    if contract["selected_songs"] != [song["song_id"] for song in songs]:
        raise ValueError("Source selected-song inventory differs from its dataset/config")
    provider = contract["provider_identities"]["local"]
    if not isinstance(provider, dict) or type(provider.get("synthetic")) is not bool:
        raise ValueError("Source local-provider identity is incomplete")
    if set(contract["audio_identities"]) != set(contract["selected_songs"]):
        raise ValueError("Source audio identity inventory differs")
    for song in songs:
        if not _same(audio_identity(contract["audio_identities"][song["song_id"]]), audio_identity(song["audio"])):
            raise ValueError("Source actual-audio identity differs from its dataset")
    parser = Path(__file__).resolve().parents[1] / "scripts_omnidance" / "exp" / "ms" / "music_planner_eval.py"
    parser_sha = file_sha(parser)
    if contract["code_sha256"].get("music_planner_eval.py") != parser_sha:
        raise ValueError("Candidate source uses a different actual local-caption parser")
    binding = dict(dataset_sha256=contract["dataset_sha256"], selected_songs=contract["selected_songs"],
        audio_identities={song["song_id"]: audio_identity(song["audio"]) for song in songs},
        local_config=local_settings(cfg), provider_identity=provider,
        local_parser_sha256=parser_sha)
    rows = []
    for song in songs:
        for window in make_windows(song["audio"]["num_samples"], song["audio"]["sample_rate"],
                                   cfg["window_seconds"], cfg["stride_seconds"]):
            for candidate in range(cfg["candidates_per_window"]):
                call_id = song["song_id"] + "/local/%04d/%02d" % (window["index"], candidate)
                rows.append(dict(song_id=song["song_id"], window=window, candidate_index=candidate, call_id=call_id))
    return cfg, songs, binding, rows


def _check_entry(entry, expected, contract, source_result, cfg):
    from .baseline import _request, _verify
    if not isinstance(entry, dict) or set(entry) != {"song_id", "window", "candidate_index", "call_id", "started", "raw"}:
        raise ValueError("Unexpected candidate-bank entry fields")
    if not _same({k: entry[k] for k in expected}, expected):
        raise ValueError("Candidate song/window/candidate index differs from its source schedule")
    started, raw = _verify(entry["started"]), _verify(entry["raw"])
    contract_sha = object_sha(contract)
    if (started.get("schema") != "choreobench.call_started.v1" or raw.get("schema") != "choreobench.call_raw.v1"
            or started.get("call_id") != entry["call_id"] or raw.get("call_id") != entry["call_id"]
            or started.get("contract_sha256") != contract_sha or raw.get("contract_sha256") != contract_sha
            or raw.get("started_sha256") != started["record_sha256"]
            or started.get("request_sha256") != object_sha(started.get("request"))
            or not _same(started.get("provider_identity"), contract["provider_identities"]["local"])
            or source_result["call_records"].get(entry["call_id"]) != raw["record_sha256"]):
        raise ValueError("Candidate raw/request/provider identity is not bound to the source journal result")
    request = started["request"]
    canonical_local_request(request)
    audio = request["audio"]
    source_audio = contract["audio_identities"][entry["song_id"]]
    window = entry["window"]
    if (type(audio["sha256"]) is not str or len(audio["sha256"]) != 64
            or any(type(audio[key]) is not int for key in ("sample_rate", "num_samples", "parent_start", "parent_end"))
            or audio["parent_audio_sha256"] != source_audio["sha256"]
            or audio["parent_start"] != window["observation_start"]
            or audio["parent_end"] != window["observation_end"]
            or audio["sample_rate"] != source_audio["sample_rate"]
            or audio["num_samples"] != window["observation_end"] - window["observation_start"]
            or audio["conversion"] != "lossless PCM16 sample slice; no padding, resampling or mixing"
            or not _same(request, _request(entry["call_id"], cfg, response_kind="local_caption", audio=audio))):
        raise ValueError("Candidate request/audio/seed/budget differs from the source contract")


def _validate_bank(data):
    from .baseline import ModelOutputError, _local_actions, _recorded_result, _verify
    _verify(data)
    required = {"schema", "source", "binding", "entries", "song_states", "synthetic", "original_collection_calls",
                "original_collection_costs", "expected_local_calls", "record_sha256"}
    if not isinstance(data, dict) or set(data) != required or data["schema"] != "choreobench.local_candidates.v1":
        raise ValueError("Unknown local candidate-bank schema/fields")
    source = data["source"]
    if not isinstance(source, dict) or set(source) != {"run_dir", "contract", "contract_file_sha256", "results", "results_file_sha256"}:
        raise ValueError("Incomplete candidate source evidence")
    contract, result = source["contract"], _verify(source["results"])
    if result.get("contract_sha256") != object_sha(contract):
        raise ValueError("Candidate source results belong to another contract")
    cfg, songs, binding, rows = _source_context(contract)
    if not _same(binding, data["binding"]) or type(data["synthetic"]) is not bool:
        raise ValueError("Candidate bank binding differs from original collection evidence")
    # Preserve source-run synthetic taint, even if only its outer planner was mock.
    if data["synthetic"] != bool(contract["synthetic"] or binding["provider_identity"]["synthetic"]):
        raise ValueError("Candidate bank changed its source synthetic status")
    if not isinstance(data["entries"], list):
        raise ValueError("Candidate entries must be an ordered list")
    entries = {}
    for entry in data["entries"]:
        if not isinstance(entry, dict) or entry.get("call_id") in entries:
            raise ValueError("Duplicate/malformed candidate entries")
        entries[entry.get("call_id")] = entry
    used, states, costs = [], [], []
    for song in songs:
        expected_rows = [row for row in rows if row["song_id"] == song["song_id"]]
        state = dict(song_id=song["song_id"], expected_local_calls=len(expected_rows), recorded_local_calls=0,
                     status="complete", first_failure_call_id=None, failure=None)
        for row in expected_rows:
            entry = entries.get(row["call_id"])
            if entry is None:
                raise ValueError("Missing local candidate without a recorded terminal model failure: " + row["call_id"])
            _check_entry(entry, row, contract, result, cfg)
            used.append(row["call_id"])
            state["recorded_local_calls"] += 1
            try:
                response = _recorded_result(entry["raw"], binding["provider_identity"], entry["started"]["request"])
                _local_actions(response["text"])
            except ModelOutputError as error:
                state.update(status="local_model_error", first_failure_call_id=row["call_id"], failure=str(error))
                costs.append(dict(copy.deepcopy(entry["raw"]["result"]["costs"]), call_id=row["call_id"]))
                break
            costs.append(dict(copy.deepcopy(response["costs"]), call_id=row["call_id"]))
        states.append(state)
    # Include all and only the source's actually observed local prefix, never a
    # convenient valid subset of a source with missing/extra local records.
    source_local_ids = {call_id for call_id in result["call_records"] if "/local/" in call_id}
    if (set(used) != set(entries) or source_local_ids != set(entries)
            or [entry["call_id"] for entry in data["entries"]] != used):
        raise ValueError("Candidate bank omits/reorders source local evidence or includes extra samples")
    if (not _same(states, data["song_states"]) or not _same(costs, data["original_collection_costs"])
            or type(data["original_collection_calls"]) is not int or data["original_collection_calls"] != len(entries)
            or type(data["expected_local_calls"]) is not int or data["expected_local_calls"] != len(rows)):
        raise ValueError("Candidate state/cost/coverage inventory differs from immutable raw evidence")
    return entries, states, costs


class CandidateBank:
    def __init__(self, data, path, bytes_sha, expected_stat):
        self.entries, self.states, self.costs = _validate_bank(data)
        self.data = data
        self.path, self.bytes_sha = Path(path).resolve(), bytes_sha
        self._stat = expected_stat
        self.assert_unchanged()
        self.identity = copy.deepcopy(data["binding"]["provider_identity"])
        self.reused = {}

    def call(self, request):
        from .baseline import _recorded_result
        self.assert_unchanged()
        entry = self.entries.get(request["call_id"])
        if entry is None or not _same(canonical_local_request(request), canonical_local_request(entry["started"]["request"])):
            raise ValueError("Reused local request differs in audio bytes/range/clock/seed/budget/identity")
        self.reused[request["call_id"]] = entry["raw"]["record_sha256"]
        return copy.deepcopy(_recorded_result(entry["raw"], self.identity, request))

    def assert_unchanged(self, full=False):
        """O(1) metadata check per window; full digest only at run boundaries."""
        if _stat(self.path) != self._stat or (full and file_sha(self.path) != self.bytes_sha):
            raise ValueError("Candidate bank bytes/metadata changed during reuse")

    def summary(self):
        return dict(schema=self.data["schema"], path=str(self.path), file_sha256=self.bytes_sha,
            bank_sha256=self.data["record_sha256"], selected_songs=self.data["binding"]["selected_songs"],
            synthetic=self.data["synthetic"], expected_local_calls=self.data["expected_local_calls"],
            recorded_local_calls=self.data["original_collection_calls"],
            failed_songs=[copy.deepcopy(s) for s in self.states if s["status"] != "complete"],
            original_collection_calls=self.data["original_collection_calls"], original_collection_costs=self.costs,
            original_provider_identity_sha256=object_sha(self.identity),
            original_collection_contract_sha256=object_sha(self.data["source"]["contract"]),
            current_native_weights_loaded_or_audited=False)


def load_candidates(path):
    before = _stat(path)
    data, bytes_sha = read_json_snapshot(path)
    return CandidateBank(data, path, bytes_sha, before)


def inspect_candidates(path):
    """Read/validate the bank only. Does not open current model/checkpoint files."""
    return load_candidates(path).summary()


def match_candidates(bank, dataset_sha, songs, inputs, config):
    expected = dict(dataset_sha256=dataset_sha, selected_songs=[s["song_id"] for s in songs],
        audio_identities={s["song_id"]: audio_identity(inputs[s["song_id"]]) for s in songs},
        local_config=local_settings(config), provider_identity=bank.identity,
        local_parser_sha256=bank.data["binding"]["local_parser_sha256"])
    if not _same(expected, bank.data["binding"]):
        raise ValueError("Candidate bank dataset/song/audio/window/seed/local configuration identity differs")
    return bank.summary()


def export_candidates(run_dir, out):
    """Export all source local observations, including terminal model failures.

    Missing/pending/infrastructure-failed local calls are never converted into
    mock candidates or automatically retried. An outer failure does not erase
    already-complete local collection evidence.
    """
    from .baseline import ModelOutputError, _Journal, _local_actions, _recorded_result, _sealed, _verify
    run_dir, out = Path(run_dir).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(str(out))
    if not (run_dir / "calls").is_dir():
        raise ValueError("Candidate export requires an existing local journal")
    with file_lock(run_dir / ".run.lock"):
        contract, contract_file_sha = read_json_snapshot(run_dir / "contract.json")
        result, results_file_sha = read_json_snapshot(run_dir / "results.json")
        _verify(result)
        cfg, songs, binding, rows = _source_context(contract)
        journal = _Journal(run_dir, object_sha(contract), cfg["max_calls"])
        if (result.get("contract_sha256") != object_sha(contract)
                or result.get("call_records") != {key: value["record_sha256"] for key, value in journal.records.items()}):
            raise ValueError("Source journal/results snapshot differs")
        entries, states, costs = [], [], []
        for song in songs:
            actual = inspect_audio(contract["audio_identities"][song["song_id"]]["path"])
            if not _same(audio_identity(actual), audio_identity(song["audio"])):
                raise ValueError("Actual source audio changed before candidate export")
            expected_rows = [row for row in rows if row["song_id"] == song["song_id"]]
            state = dict(song_id=song["song_id"], expected_local_calls=len(expected_rows), recorded_local_calls=0,
                         status="complete", first_failure_call_id=None, failure=None)
            for row in expected_rows:
                call_id = row["call_id"]
                if call_id not in journal.started or call_id not in journal.records:
                    raise ValueError("Missing/pending local collection; no export or resampling: " + call_id)
                entry = dict(row, started=journal.started[call_id], raw=journal.records[call_id])
                _check_entry(entry, row, contract, result, cfg)
                audio = entry["started"]["request"]["audio"]
                derived = inspect_audio(audio["path"])
                if (not _same(audio_identity(derived), audio_identity(audio)) or derived["pcm"] !=
                        actual["pcm"][2 * row["window"]["observation_start"]:2 * row["window"]["observation_end"]]):
                    raise ValueError("Original expert observation bytes are not the declared source PCM slice")
                entries.append(entry)
                state["recorded_local_calls"] += 1
                try:
                    response = _recorded_result(entry["raw"], binding["provider_identity"], entry["started"]["request"])
                    _local_actions(response["text"])
                except ModelOutputError as error:
                    state.update(status="local_model_error", first_failure_call_id=call_id, failure=str(error))
                    costs.append(dict(copy.deepcopy(entry["raw"]["result"]["costs"]), call_id=call_id))
                    break
                costs.append(dict(copy.deepcopy(response["costs"]), call_id=call_id))
            states.append(state)
        bank = _sealed(dict(schema="choreobench.local_candidates.v1",
            source=dict(run_dir=str(run_dir), contract=contract, contract_file_sha256=contract_file_sha,
                        results=result, results_file_sha256=results_file_sha),
            binding=binding, entries=entries, song_states=states,
            synthetic=bool(contract["synthetic"] or binding["provider_identity"]["synthetic"]),
            original_collection_calls=len(entries), original_collection_costs=costs, expected_local_calls=len(rows)))
        _validate_bank(bank)
        write_json(out, bank)
    return inspect_candidates(out)
