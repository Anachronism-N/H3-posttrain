"""One-shot N4 GPU2/3 v4 handoff; reuse frozen v1 safety, never patch it."""
import argparse
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
import caption_light_to_local_transition_v1 as frozen

RT = Path(__file__).resolve().parent
WRAPPER = Path(__file__).resolve()
OLD_PHASE = 'caption_two_stage_live_n4c4_v4'
OLD_CONFIG = RT / (OLD_PHASE + '_launch.json')
NEW_PHASE = 'caption_two_stage_local_n4c4_v1'
FROZEN_SHA = '0b53113c5e37f114a219ded9b1b49888d53139c5e9776fb4159ef7053c6cd5ee'
LOCAL_SHA = '74dae116f4db431e6fed16fc86da0d8f41af1cbd9316f6052b54ef9433a14b85'
OLD_PRODUCERS = {
    str(RT / 'caption_two_stage_live_manager_v3.py'): '2076912d5b9b57c98e815affbbbe5f5d47e080e3493f9cdcef12071870b37687',
    str(RT / 'caption_two_stage_live_manager_v4.py'): '92e94f9ce90a830552eeab35818e6a3603d8da442df31d870f8e7aee5dad9a5b',
    str(RT / 'caption_two_stage_live_manager_v5.py'): '18402b5edfeb121adb6e83ae30d679bf7290af77f03dfc799ebb7e2072ae1173',
}
m = frozen.m
need = frozen.need


def old_manager(old, old_path):
    producer = old['producer']
    path = producer['path']
    need(path in OLD_PRODUCERS and producer == dict(path=path, sha256=OLD_PRODUCERS[path])
         and m.ref(path) == producer, 'Old producer path/SHA not a frozen v3/v4/v5 identity')
    need(Path(path).name == 'caption_two_stage_live_manager_v4.py', 'Only this v4 lane is supported')
    need(Path(old_path).resolve() == OLD_CONFIG and old['config_path'] == str(OLD_CONFIG)
         and old['phase'] == OLD_PHASE and old['node'] == 'N4' and old['gpus'] == [2, 3]
         and old['draft_out'] == str(m.BAR / OLD_PHASE)
         and old['queue'] == str(m.BAR / 'queues' / OLD_PHASE), 'Wrong fixed N4 GPU2/3 old lane')
    return Path(path)


def capacity():
    budget = frozen.capacity()
    return dict(budget, existing_copy_threads=4, added_copy_threads=1, total_copy_threads=5,
                existing_node_copy_threads=2, added_node_copy_threads=1, total_node_copy_threads=3)


def check_copy_budget(budget):
    need(all(budget.get(key) == value for key, value in dict(
        existing_copy_threads=4, added_copy_threads=1, total_copy_threads=5,
        existing_node_copy_threads=2, added_node_copy_threads=1, total_node_copy_threads=3,
        source_batch_size=3, max_inflight_batches=2).items()), 'Copy concurrency budget changed')


def check_native_sources(cfg, config_path):
    original = m.base.check_sources
    check = types.FunctionType(original.__code__,
        dict(original.__globals__, selected_sources=m.selected), original.__name__,
        original.__defaults__, original.__closure__)
    check(cfg, str(config_path), m.base.sha(config_path), m.BASE_SHA)


def lane_functions(old, old_path):
    manager = old_manager(old, old_path)
    need(m.base.sha(frozen.SELF) == FROZEN_SHA and m.base.sha(m.SELF) == LOCAL_SHA,
         'Frozen transition/local manager changed')
    scope = dict(vars(frozen), LIGHT_MANAGER=manager)
    for name, function in vars(frozen).items():
        if isinstance(function, types.FunctionType) and function.__globals__ is vars(frozen):
            scope[name] = types.FunctionType(function.__code__, scope, function.__name__,
                                            function.__defaults__, function.__closure__)
    generator = frozen.handoff_lock.__wrapped__
    scope['handoff_lock'] = frozen.contextmanager(types.FunctionType(
        generator.__code__, scope, generator.__name__, generator.__defaults__, generator.__closure__))
    native_receipts = scope['check_old_receipts']
    native_bindings = scope['code_bindings']
    native_park = scope['park_old']
    native_write = scope['write_once']

    def receipts(cfg, path, admission, live):
        need(old_manager(cfg, path) == manager, 'Old producer changed after binding')
        return native_receipts(cfg, path, admission, live)

    def bindings(cfg):
        need(old_manager(cfg, old_path) == manager, 'Old producer changed after binding')
        result = native_bindings(cfg)
        result[str(WRAPPER)] = m.base.sha(WRAPPER)
        return result

    def write(path, value):
        if Path(path) == RT / (OLD_PHASE + '_light_to_local_handoff_claim.json'):
            value = dict(value, frozen_transition=value['transition'], transition=m.ref(WRAPPER))
        return native_write(path, value)

    def park(*args):
        check_copy_budget(scope['capacity']())
        return native_park(*args)

    def final_check(cfg, config_path, expected, proofs, budgets):
        scope['check_files'](proofs)
        scope['old_tree_absent'](expected)
        check_copy_budget(budgets)
        check_copy_budget(scope['capacity']())
        previous = frozen.load(cfg['previous_config']['path'])
        queue = Path(previous['queue'])
        parent, _ = receipts(previous, Path(cfg['previous_config']['path']),
                             frozen.load(queue / 'admission.json'), frozen.load(queue / 'live.json'))
        scope['safe_parent_wait'](previous, parent)
        check_native_sources(cfg, config_path)
        m.validate(cfg)

    scope.update(check_old_receipts=receipts, code_bindings=bindings, write_once=write,
                 park_old=park, capacity=capacity, final_check=final_check)
    return scope


def run(old_path=OLD_CONFIG, name=NEW_PHASE, prepare_only=False):
    need(name == NEW_PHASE, 'Only the fixed new N4 GPU2/3 local lane is supported')
    old_path = Path(old_path).resolve()
    old = frozen.load(old_path)
    scope = lane_functions(old, old_path)
    return scope['run'](old_path, name, prepare_only)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-config', default=str(OLD_CONFIG))
    parser.add_argument('--name', default=NEW_PHASE)
    parser.add_argument('--prepare-only', action='store_true',
                        help='Still parks the old lane; does not exec the new manager')
    args = parser.parse_args()
    run(args.old_config, args.name, args.prepare_only)


if __name__ == '__main__':
    main()
