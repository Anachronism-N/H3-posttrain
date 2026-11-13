"""Opt-in bounded composition control, shared by ReAct and choreography.

This changes execution policy, not model weights or semantic quality validation.
Full tool observations remain in immutable step records. No historical run is
rewritten or retried by this controller.
"""
import copy

from .composition_workflows import CompositionState
from .io import object_sha
from .schema import keys, text


NO_PROGRESS_LIMIT = 3
TOOL_PROMPT = (
    'The earlier block-only output instruction is superseded. '
    'Return exactly {tool,proposal,reason}, choosing only available_tools. '
    'Use block_contract for each proposal and its event fields. '
    'propose_block stores a draft; revise_block changes it at most once per unit. '
    'submit_block validates and commits a supplied proposal in one decision. '
    'Changing a stored draft through submit_block also spends its one revision. '
    'commit_block validates and commits the stored draft with proposal:null. '
    'An unchanged revision requests retention: the controller commits it only if structurally valid. '
    'Checks validate structure, never music, aesthetics or physical feasibility. '
    'Committed blocks cannot change. Read tools do not listen to audio again. '
    'Prefer submit_block for a ready proposal; optional inspections must leave '
    'one model call for every remaining block. All committed blocks finish automatically. '
    'A new reason without new content is not progress. Use no extra fields.'
)


class CompositionStateV2(CompositionState):
    def __init__(self, source, outline):
        super().__init__(source, outline)
        self.remaining_calls = None
        self.no_progress_streak = 0
        self.no_progress_steps = 0
        self.automatic_commits = 0

    def progress_signature(self):
        return object_sha(dict(committed_units=self.index, draft=self.draft))

    def draft_is_valid(self):
        if self.draft is None:
            return False
        try:
            self.validate(self.draft)
        except (ValueError, TypeError, KeyError):
            return False
        return True

    def available_tools(self, remaining_calls=None):
        remaining = len(self.source['units']['units']) - self.index
        if remaining == 0:
            return ['finish']
        calls = self.remaining_calls if remaining_calls is None else remaining_calls
        valid = self.draft_is_valid()
        if self.draft is not None and self.revisions_used >= 1 and not valid:
            return []
        if calls is not None and calls <= remaining:
            # One decision must finish this unit; a draft is not a committed unit.
            return ['commit_block'] if valid else ['submit_block']
        tools = ['read_evidence', 'read_history']
        if self.draft is None:
            return tools + ['propose_block', 'submit_block']
        tools += ['check_block']
        if valid:
            tools += ['commit_block']
        if self.revisions_used < 1:
            tools += ['revise_block', 'submit_block']
        return tools

    def _store(self, proposal):
        keys(proposal, ('unit_id', 'events'), where='block draft')
        units = self.source['units']['units']
        if self.index == len(units) or proposal['unit_id'] != units[self.index]['unit_id']:
            raise ValueError('Draft targets a different unit')
        self.draft = copy.deepcopy(proposal)

    def _commit(self, tool, automatic_reason=None):
        # validate() builds a separate plan. No history mutates on failure.
        events, plan = self.validate(self.draft)
        unit_id = self.draft['unit_id']
        self.plan = plan
        self.index += 1
        self.draft = None
        self.revisions_used = 0
        observation = dict(tool=tool, status='committed', unit_id=unit_id,
                           quality_verified=False)
        if automatic_reason:
            self.automatic_commits += 1
            observation['automatic_commit_reason'] = automatic_reason
        return observation

    def execute(self, decision):
        keys(decision, ('tool', 'proposal', 'reason'), where='composition tool')
        text(decision['reason'], 'tool reason')
        if len(decision['reason']) > 1200:
            raise ValueError('Tool reason budget exceeded')
        tool, proposal = decision['tool'], decision['proposal']
        if tool not in self.available_tools():
            raise ValueError('Tool is not allowed in the current controller state: ' + str(tool))
        if tool not in ('propose_block', 'revise_block', 'submit_block') and proposal is not None:
            raise ValueError('Read/check/commit/finish cannot mutate draft')
        if tool == 'submit_block':
            if self.draft is not None:
                # Count attempts, even if malformed, rather than letting submit
                # provide unlimited revisions under a different tool name.
                self.revisions_used += 1
            self._store(proposal)
            return self._commit(tool)
        if tool == 'revise_block':
            self.revisions_used += 1
            unchanged = proposal == self.draft
            self._store(proposal)
            if unchanged:
                return self._commit(tool, 'unchanged_revision')
            return dict(tool=tool, status='drafted', draft_sha256=object_sha(self.draft),
                        revisions_used=self.revisions_used, revision_limit=1, quality_verified=False)
        if tool == 'commit_block':
            return self._commit(tool)
        return super().execute(decision)

    def after_decision(self, before):
        """Return a visible controller commit or a stop signal after idle steps."""
        if self.progress_signature() != before:
            self.no_progress_streak = 0
            return None
        self.no_progress_streak += 1
        self.no_progress_steps += 1
        if self.no_progress_streak < NO_PROGRESS_LIMIT:
            return None
        if self.draft_is_valid():
            result = self._commit('controller_commit', 'no_content_progress')
            self.no_progress_streak = 0
            return result
        return dict(tool='controller_stop', status='stalled_no_content_progress',
                    quality_verified=False)


def observation_context(observations):
    """Expose fresh evidence once; never resend full history snapshots.

    read_evidence may reveal future candidates not in the current block, so its
    latest response must remain visible. Compact it only after another decision.
    """
    result = []
    for index, original in enumerate(observations[-2:]):
        fresh = index == len(observations[-2:]) - 1
        item = copy.deepcopy(original)
        history = item.pop('committed_plan', None)
        if history is not None:
            item['history_reference'] = dict(sha256=object_sha(history),
                                            events=len(history.get('events', [])),
                                            current_history_in='committed_plan')
        if not fresh:
            for name in ('candidates', 'units'):
                if name in item:
                    value = item.pop(name)
                    item[name + '_reference'] = dict(sha256=object_sha(value),
                                                     source='immutable_step_observation')
        result.append(item)
    return result


def choreography_context(state):
    units = state.source['units']['units']
    unit = units[state.index] if state.index < len(units) else None
    current_observations = []
    for observation in reversed(state.observations):
        if observation.get('status') == 'committed':
            break
        current_observations.append(observation)
    rejections = [o for o in reversed(current_observations) if o.get('status') == 'rejected']
    return dict(
        target_unit=None if unit is None else {k: unit[k] for k in ('unit_id', 'start', 'end')},
        last_committed_actions=[dict(event_id=e['event_id'], action=e['action'])
                                for e in state.plan['events'][-3:]],
        unresolved_structure_issues=rejections[-3:],
        active_constraints=[state.source['original']['brief'], 'Solo motion only; no appearance/background',
                            'Only current uncommitted block may change',
                            'Original schedule is design timing, not observed action boundary'],
        draft_state='absent' if state.draft is None else 'uncommitted',
    )
