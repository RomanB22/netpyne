"""
Helpers for a CoreNEURON report-based LFP backend.

This backend replaces NetPyNE's current CoreNEURON ``Vector.record(seg._ref_i_membrane_)``
path with CoreNEURON file-mode reporting:

1. NetPyNE writes ``sim.conf`` and ``report.conf``
2. CoreNEURON emits a SONATA ``i_membrane`` compartment report
3. NetPyNE parses that report and reuses the existing transfer-resistance LFP code
"""

from pathlib import Path
import json
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


def use_report_imem_backend(cfg):
    """Return True when the report-based CoreNEURON LFP path is selected."""

    return bool(getattr(cfg, 'coreneuron', False)) and getattr(cfg, 'coreneuronLFPBackend', 'vector') == 'report_imem'


def _default_report_dir():
    from .. import sim

    if getattr(sim.cfg, 'coreneuronReportDir', ''):
        return Path(sim.cfg.coreneuronReportDir)

    save_root = Path(sim.cfg.saveFolder) if getattr(sim.cfg, 'saveFolder', '') else Path('.')
    base_name = Path(getattr(sim.cfg, 'filename', '') or getattr(sim.cfg, 'simLabel', '') or 'model_output').name
    return save_root / f'{base_name}_coreneuron_reports'


def init_report_state():
    """Initialise and cache report-backend state on the shared ``sim`` module."""

    from .. import sim

    state = getattr(sim, '_coreneuron_report_state', None)
    if state is not None:
        return state

    report_dir = _default_report_dir()
    report_name = getattr(sim.cfg, 'coreneuronReportName', 'imembrane')
    report_filename = report_name if str(report_name).endswith('.h5') else f'{report_name}.h5'
    report_population = getattr(sim.cfg, 'coreneuronReportPopulation', '')
    target_name = report_population.strip('/').split('/')[-1] if report_population else 'NetPyNECells'

    state = {
        'backend': getattr(sim.cfg, 'coreneuronLFPBackend', 'vector'),
        'report_dir': str(report_dir),
        'manifest_path': str(report_dir / 'netpyne_corenrn_report_manifest.json'),
        'sim_conf_path': str(report_dir / 'sim.conf'),
        'report_conf_path': str(report_dir / 'report.conf'),
        'report_data_path': str(report_dir / report_filename),
        'datpath': str(report_dir / 'coreneuron_input'),
        'outpath': str(report_dir),
        'report_name': report_name,
        'report_filename': report_filename,
        'report_population': report_population,
        'target_name': target_name,
        'mapping_name': 'all',
        'record_step': float(sim.cfg.recordStep),
        'duration': float(sim.cfg.duration),
        'nsites': len(getattr(sim.cfg, 'recordLFP', [])),
        'cells': {},
        'segment_order': {},
        'total_cells': 0,
        'total_segments': 0,
        'config_files_written': False,
        'report_buffer_size': 8,
        'file_mode_armed': False,
        'mapping_available': None,
        'mapping_registered': 0,
        'mapping_error': '',
    }

    sim._coreneuron_report_state = state
    return state


def _iter_cell_segments(cell):
    for sec_index, sec in enumerate(list(cell.secs.values())):
        hSec = sec['hObj']
        for seg in hSec:
            yield sec_index, seg


def _build_section_segment_mapping(cell):
    section_ids = []
    segment_ids = []
    order = []

    for sec_index, sec in enumerate(list(cell.secs.values())):
        hSec = sec['hObj']
        for seg_index, _seg in enumerate(hSec):
            section_ids.append(int(sec_index))
            segment_ids.append(int(seg_index))
            order.append((int(sec_index), int(seg_index)))

    return order, section_ids, segment_ids


def _register_cell_mapping(gid, section_ids, segment_ids, state):
    from .. import sim

    if state['mapping_available'] is None:
        state['mapping_available'] = hasattr(sim.pc, 'nrnbbcore_register_mapping')
        if not state['mapping_available']:
            state['mapping_error'] = (
                "The installed NEURON ParallelContext does not expose "
                "nrnbbcore_register_mapping(), so CoreNEURON report metadata files "
                "(gid_3.dat) cannot be generated."
            )

    if not state['mapping_available']:
        return False

    from neuron import h

    sec_vec = h.Vector(len(section_ids))
    seg_vec = h.Vector(len(segment_ids))
    sec_vec.from_python(section_ids)
    seg_vec.from_python(segment_ids)
    try:
        sim.pc.nrnbbcore_register_mapping(int(gid), state['mapping_name'], sec_vec, seg_vec)
    except Exception as exc:
        state['mapping_error'] = (
            f"nrnbbcore_register_mapping failed for gid {int(gid)}: {exc}"
        )
        raise RuntimeError(state['mapping_error']) from exc
    state['mapping_registered'] += 1
    return True


def _validate_mapping_support(state):
    if state['mapping_available'] is False:
        raise RuntimeError(
            "cfg.coreneuronLFPBackend='report_imem' requires "
            "ParallelContext.nrnbbcore_register_mapping() so CoreNEURON can write "
            "gid_3.dat mapping files. "
            f"{state['mapping_error']}"
        )

    missing = [gid for gid, info in state['cells'].items() if not info.get('mapping_registered')]
    if missing:
        preview = ', '.join(str(gid) for gid in missing[:8])
        suffix = '...' if len(missing) > 8 else ''
        raise RuntimeError(
            "CoreNEURON report mapping registration did not complete for all local gids. "
            f"Missing mappings for {len(missing)} gids: {preview}{suffix}"
        )


def register_cell_for_imem_report(cell):
    """Record lightweight per-cell metadata for future report-file generation."""

    state = init_report_state()
    gid = int(cell.gid)
    if gid in state['cells']:
        return

    nseg = int(cell.getNumberOfSegments())
    segment_order, section_ids, segment_ids = _build_section_segment_mapping(cell)
    mapping_registered = _register_cell_mapping(gid, section_ids, segment_ids, state)
    state['cells'][gid] = {
        'gid': gid,
        'pop': cell.tags.get('pop', ''),
        'nseg': nseg,
        'mapping_name': state['mapping_name'],
        'mapping_registered': mapping_registered,
    }
    state['segment_order'][gid] = segment_order
    state['total_cells'] += 1
    state['total_segments'] += nseg


def _quote_path(path):
    return "'" + str(Path(path).expanduser().resolve()).replace("'", "\\'") + "'"


def _write_sim_conf(state):
    from .. import sim

    lines = [
        f"outpath={_quote_path(state['outpath'])}",
        f"datpath={_quote_path(state['datpath'])}",
        f'tstop={float(sim.cfg.duration)}',
        f'dt={float(sim.cfg.dt)}',
        f'prcellgid=-1',
        f'celsius={float(sim.cfg.hParams["celsius"])}',
        f'voltage={float(sim.cfg.hParams["v_init"])}',
        f'cell-permute={2 if getattr(sim.cfg, "gpu", False) else 0}',
        f'mpi={1 if sim.nhosts > 1 else 0}',
        f'report-conf={_quote_path(state["report_conf_path"])}',
    ]
    Path(state['sim_conf_path']).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _write_binary_i32(fh, values):
    arr = np.asarray(values, dtype=np.int32)
    fh.write(arr.astype('<i4', copy=False).tobytes())
    fh.write(b'\n')


def _gather_root_lists(local_values):
    from .. import sim

    if sim.nhosts <= 1:
        return [list(local_values)]

    data = [None] * sim.nhosts
    data[0] = list(local_values)
    gathered = sim.pc.py_alltoall(data)
    sim.pc.barrier()
    return gathered if sim.rank == 0 else None


def _gather_root_value(local_value):
    from .. import sim

    if sim.nhosts <= 1:
        return [local_value]

    data = [None] * sim.nhosts
    data[0] = local_value
    gathered = sim.pc.py_alltoall(data)
    sim.pc.barrier()
    return gathered if sim.rank == 0 else None


def _write_report_conf(state):
    """Write a simple compartment report for i_membrane over all registered gids."""

    from .. import sim

    gathered = _gather_root_lists(sorted(state['cells']))
    if sim.rank != 0:
        return

    global_gids = sorted(int(gid) for gids in gathered if gids for gid in gids)
    state['report_gids'] = global_gids

    # The official docs list the metadata fields in one order, but the example uses
    # ``... dt start end num_gids buffer_size scaling``. Follow the example, which
    # is consistent with compartment reports that provide one gid list.
    metadata = (
        f'{state["report_filename"]} '
        f'{state["target_name"]} '
        f'compartment '
        f'i_membrane '
        f'nA '
        f'SONATA '
        f'all '
        f'{state["mapping_name"]} '
        f'{state["record_step"]} '
        f'0.0 '
        f'{state["duration"]} '
        f'{len(global_gids)} '
        f'{state["report_buffer_size"]} '
        f'none'
    )

    with Path(state['report_conf_path']).open('wb') as fh:
        fh.write(b'1\n')
        fh.write((metadata + '\n').encode('utf-8'))
        _write_binary_i32(fh, global_gids)


def _write_manifest(state):
    manifest_cells = state.get('manifest_cells', state['cells'])
    manifest_totals = state.get(
        'manifest_totals',
        {'cells': state['total_cells'], 'segments': state['total_segments']},
    )
    manifest = {
        'backend': state['backend'],
        'status': 'configured',
        'report': {
            'name': state['report_name'],
            'filename': state['report_filename'],
            'population': state['report_population'],
            'target_name': state['target_name'],
            'mapping_name': state['mapping_name'],
            'variable': 'i_membrane',
            'unit': 'nA',
            'report_type': 'compartment',
            'record_step_ms': state['record_step'],
            'duration_ms': state['duration'],
            'nsites': state['nsites'],
            'data_path': state['report_data_path'],
            'buffer_size': state['report_buffer_size'],
            'num_gids': len(state.get('report_gids', [])),
        },
        'paths': {
            'report_dir': state['report_dir'],
            'manifest_path': state['manifest_path'],
            'sim_conf_path': state['sim_conf_path'],
            'report_conf_path': state['report_conf_path'],
            'datpath': state['datpath'],
            'outpath': state['outpath'],
        },
        'mapping': {
            'available': state.get('mapping_available'),
            'registered_cells': state.get('mapping_registered', 0),
            'error': state.get('mapping_error', ''),
        },
        'cells': manifest_cells,
        'totals': manifest_totals,
    }

    Path(state['manifest_path']).write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _remove_stale_report_output(state):
    report_file = Path(state['report_data_path'])
    if report_file.exists():
        report_file.unlink()


def _write_runtime_configs(state):
    from .. import sim

    lines = [
        state['report_dir'],
        state['datpath'],
        state['outpath'],
    ]
    for path in lines:
        Path(path).mkdir(parents=True, exist_ok=True)

    gathered_cells = _gather_root_value(state['cells'])
    _write_report_conf(state)

    if sim.rank == 0:
        if gathered_cells:
            merged_cells = {}
            for cell_map in gathered_cells:
                if cell_map:
                    merged_cells.update(cell_map)
            state['manifest_cells'] = merged_cells
            state['manifest_totals'] = {
                'cells': len(merged_cells),
                'segments': sum(cell_info['nseg'] for cell_info in merged_cells.values()),
            }

        _write_sim_conf(state)
        if Path(state['manifest_path']).parent.exists():
            _write_manifest(state)
        _remove_stale_report_output(state)

    state['config_files_written'] = True
    sim.pc.barrier()


def finalize_report_setup():
    """Write manifest and runtime config files for the report-based backend."""

    state = init_report_state()
    if state['config_files_written']:
        return
    _validate_mapping_support(state)
    _write_runtime_configs(state)

    from .. import sim

    if sim.rank == 0:
        print(f'  Wrote CoreNEURON report-backend config to {state["report_dir"]}')


def prepare_report_run():
    """Configure CoreNEURON file mode so ``pc.psolve`` emits a SONATA i_membrane report."""

    state = init_report_state()
    if not state['config_files_written']:
        finalize_report_setup()

    from neuron import coreneuron

    if not hasattr(coreneuron, 'file_mode'):
        raise AttributeError("The installed neuron.coreneuron module does not expose file_mode.")

    state['previous_file_mode'] = getattr(coreneuron, 'file_mode', False)
    coreneuron.file_mode = True
    sim_conf_path = _normalise_path(state['sim_conf_path'])
    datpath = _normalise_path(state['datpath'])

    if not hasattr(coreneuron, 'nrncore_arg'):
        raise AttributeError(
            "The installed neuron.coreneuron module does not expose nrncore_arg, "
            "so NetPyNE cannot direct CoreNEURON file mode to the configured datpath."
        )

    state['previous_nrncore_arg'] = coreneuron.nrncore_arg

    def nrncore_arg_with_netpyne_config(
        tstop,
        _orig=state['previous_nrncore_arg'],
        _datpath=datpath,
        _sim_conf_path=sim_conf_path,
        _use_read_config=not hasattr(coreneuron, 'sim_config'),
    ):
        arg = _orig(tstop)
        if '--datpath' not in arg:
            arg = f'{arg} --datpath {_datpath}'
        if _use_read_config and '--read-config' not in arg:
            arg = f'{arg} --read-config {_sim_conf_path}'
        return arg

    coreneuron.nrncore_arg = nrncore_arg_with_netpyne_config

    from .. import sim

    if hasattr(coreneuron, 'sim_config'):
        state['previous_sim_config'] = getattr(coreneuron, 'sim_config', '')
        coreneuron.sim_config = sim_conf_path
        state['sim_config_mode'] = 'attribute'
    else:
        state['sim_config_mode'] = 'nrncore_arg'

    state['file_mode_armed'] = True
    if sim.rank == 0:
        print(f'  Passing CoreNEURON datpath via --datpath {datpath}.')
        if state['sim_config_mode'] == 'nrncore_arg':
            print('  CoreNEURON sim_config attribute not available; passing sim.conf via --read-config.')


def cleanup_report_run():
    state = init_report_state()
    if not state.get('file_mode_armed'):
        return

    from neuron import coreneuron

    coreneuron.file_mode = state.get('previous_file_mode', False)
    if state.get('sim_config_mode') == 'attribute' and hasattr(coreneuron, 'sim_config'):
        coreneuron.sim_config = state.get('previous_sim_config', '')
    if hasattr(coreneuron, 'nrncore_arg') and 'previous_nrncore_arg' in state:
        coreneuron.nrncore_arg = state.get('previous_nrncore_arg', coreneuron.nrncore_arg)
    state['file_mode_armed'] = False


def _require_h5py():
    if h5py is None:
        raise ImportError(
            "h5py is required for cfg.coreneuronLFPBackend='report_imem'. "
            "Install h5py in the Python environment used by NetPyNE."
        )


def _normalise_path(path):
    return str(Path(path).expanduser().resolve())


def _is_report_group(obj):
    return isinstance(obj, h5py.Group) and 'data' in obj and 'mapping' in obj


def _collect_report_groups(group, prefix=''):
    groups = []
    if _is_report_group(group):
        groups.append((prefix or '/', group))
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            child_prefix = f'{prefix}/{key}' if prefix else f'/{key}'
            groups.extend(_collect_report_groups(obj, child_prefix))
    return groups


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_attr(value.item())
    return value


def _find_population_group(h5, population_name=''):
    candidates = _collect_report_groups(h5)
    if not candidates:
        raise ValueError('No SONATA element/compartment report groups were found in the HDF5 file.')

    if population_name:
        normalized = population_name.strip('/')
        for path, group in candidates:
            if path.strip('/') == normalized or path.strip('/').endswith('/' + normalized):
                return path, group
        raise ValueError(
            f"Requested SONATA report population '{population_name}' was not found. "
            f"Available groups: {[path for path, _ in candidates]}"
        )

    if len(candidates) == 1:
        return candidates[0]

    # Prefer the common SONATA layout /report/<population> when no explicit population was provided.
    report_candidates = [(path, group) for path, group in candidates if path.count('/') == 2 and path.startswith('/report/')]
    if len(report_candidates) == 1:
        return report_candidates[0]

    return sorted(candidates, key=lambda item: item[0])[0]


def _read_time_info(report_group, nsteps):
    mapping = report_group['mapping']
    time_ds = mapping.get('time')

    if time_ds is None:
        attrs = report_group.attrs
        tstart = float(_decode_attr(attrs.get('tstart', 0.0)))
        dt = float(_decode_attr(attrs.get('dt', 0.0)))
        tstop = float(_decode_attr(attrs.get('tstop', tstart + dt * max(nsteps - 1, 0))))
        return {'tstart': tstart, 'tstop': tstop, 'dt': dt}

    time_arr = np.asarray(time_ds)
    if time_arr.size >= 3:
        tstart = float(time_arr.flat[0])
        tstop = float(time_arr.flat[1])
        dt = float(time_arr.flat[2])
    elif time_arr.size == 2:
        tstart = float(time_arr.flat[0])
        tstop = float(time_arr.flat[1])
        dt = (tstop - tstart) / max(nsteps - 1, 1) if nsteps > 1 else 0.0
    elif time_arr.size == 1:
        tstart = float(time_arr.flat[0])
        dt = float(_decode_attr(time_ds.attrs.get('dt', 0.0)))
        tstop = tstart + dt * max(nsteps - 1, 0)
    else:
        raise ValueError('SONATA report time dataset is empty.')

    return {'tstart': tstart, 'tstop': tstop, 'dt': dt}


def _load_mapping_arrays(report_group):
    mapping = report_group['mapping']

    node_key = 'node_ids' if 'node_ids' in mapping else 'gids' if 'gids' in mapping else None
    if node_key is None:
        raise ValueError("SONATA report mapping is missing 'node_ids'/'gids'.")

    pointer_key = 'index_pointer' if 'index_pointer' in mapping else 'offsets' if 'offsets' in mapping else None
    if pointer_key is None:
        raise ValueError("SONATA report mapping is missing 'index_pointer'/'offsets'.")

    node_ids = np.asarray(mapping[node_key], dtype=np.int64)
    index_pointer = np.asarray(mapping[pointer_key], dtype=np.int64)
    element_ids = np.asarray(mapping['element_ids'], dtype=np.int64) if 'element_ids' in mapping else None
    element_pos = np.asarray(mapping['element_pos'], dtype=np.float64) if 'element_pos' in mapping else None

    return {
        'node_ids': node_ids,
        'index_pointer': index_pointer,
        'element_ids': element_ids,
        'element_pos': element_pos,
    }


def _build_node_ranges(node_ids, index_pointer, ncols):
    if len(index_pointer) == len(node_ids) + 1:
        starts = index_pointer[:-1]
        ends = index_pointer[1:]
    elif len(index_pointer) == len(node_ids):
        starts = index_pointer
        ends = np.concatenate((index_pointer[1:], [ncols]))
    else:
        raise ValueError(
            f'Unexpected SONATA index_pointer length {len(index_pointer)} for {len(node_ids)} node ids.'
        )

    ranges = {}
    for gid, start, end in zip(node_ids, starts, ends):
        ranges[int(gid)] = (int(start), int(end))
    return ranges


def _load_report_metadata(report_path, population_name=''):
    _require_h5py()

    with h5py.File(report_path, 'r') as h5:
        path, report_group = _find_population_group(h5, population_name)
        data = report_group['data']
        if data.ndim != 2:
            raise ValueError(f"SONATA report dataset '{path}/data' is expected to be 2D, got ndim={data.ndim}.")

        mapping = _load_mapping_arrays(report_group)
        ranges = _build_node_ranges(mapping['node_ids'], mapping['index_pointer'], data.shape[1])
        time = _read_time_info(report_group, data.shape[0])

        return {
            'report_path': report_path,
            'population_path': path,
            'data_shape': tuple(int(x) for x in data.shape),
            'time': time,
            'ranges': ranges,
            'mapping': mapping,
        }


def _read_gid_imem_matrix(report_path, population_name, gid):
    _require_h5py()

    with h5py.File(report_path, 'r') as h5:
        _, report_group = _find_population_group(h5, population_name)
        data = report_group['data']
        mapping = _load_mapping_arrays(report_group)
        ranges = _build_node_ranges(mapping['node_ids'], mapping['index_pointer'], data.shape[1])

        if int(gid) not in ranges:
            return None

        start, end = ranges[int(gid)]
        if end < start:
            raise ValueError(f'Invalid SONATA index range for gid {gid}: ({start}, {end})')

        return np.asarray(data[:, start:end], dtype=np.float32).T  # (nseg, nsteps)


def _copy_with_pad(src_matrix, nsteps):
    dst = np.zeros((src_matrix.shape[0], nsteps), dtype=np.float32)
    ncopy = min(src_matrix.shape[1], nsteps)
    if ncopy:
        dst[:, :ncopy] = src_matrix[:, :ncopy]
    return dst


def _round_seg_pos(value):
    return round(float(value), 9)


def _reorder_im_report_if_needed(gid, im_report, start, end, metadata, state):
    element_ids = metadata['mapping'].get('element_ids')
    element_pos = metadata['mapping'].get('element_pos')
    expected_order = state.get('segment_order', {}).get(gid)

    if element_ids is None or element_pos is None or not expected_order:
        return im_report

    report_ids = np.asarray(element_ids[start:end], dtype=np.int64)
    report_pos = np.asarray(element_pos[start:end], dtype=np.float64)
    if len(report_ids) != im_report.shape[0] or len(report_pos) != im_report.shape[0]:
        return im_report

    report_keys = [(int(sec_id), _round_seg_pos(seg_x)) for sec_id, seg_x in zip(report_ids, report_pos)]
    expected_keys = [(int(sec_id), _round_seg_pos(seg_x)) for sec_id, seg_x in expected_order]

    if report_keys == expected_keys:
        return im_report

    index_by_key = {key: idx for idx, key in enumerate(report_keys)}
    if len(index_by_key) != len(report_keys):
        return im_report
    if any(key not in index_by_key for key in expected_keys):
        return im_report

    reorder = [index_by_key[key] for key in expected_keys]
    return im_report[reorder, :]


def calculate_lfp_from_report():
    """Read a SONATA i_membrane report and reconstruct NetPyNE LFP/iMembrane/dipole arrays."""

    from .. import sim

    state = init_report_state()
    report_path = _normalise_path(state['report_data_path'])
    report_file = Path(report_path)
    if not report_file.exists():
        raise FileNotFoundError(
            f"CoreNEURON report-based backend expected SONATA report file at '{report_path}', but it does not exist."
        )

    metadata = _load_report_metadata(report_path, state.get('report_population', ''))
    state['loaded_population_path'] = metadata['population_path']
    state['loaded_report_shape'] = metadata['data_shape']
    state['loaded_report_time'] = metadata['time']
    time_info = metadata['time']

    nsteps = sim.simData['LFP'].shape[0] if sim.cfg.recordLFP else None
    if nsteps is None and sim.cfg.saveIMembrane and sim.simData['iMembrane']:
        sample_gid = next(iter(sim.simData['iMembrane']))
        nsteps = sim.simData['iMembrane'][sample_gid].shape[0]
    if nsteps is None and sim.cfg.recordDipole:
        nsteps = sim.simData['dipoleSum'].shape[0]
    if nsteps is None:
        return

    if sim.cfg.recordTime:
        t_data = sim.simData.get('t', None)
        if t_data is None or len(t_data) == 0:
            sim.simData['t'] = np.arange(nsteps, dtype=np.float64) * float(time_info['dt']) + float(time_info['tstart'])

    processed = 0
    missing = []

    with h5py.File(report_path, 'r') as h5:
        _, report_group = _find_population_group(h5, state.get('report_population', ''))
        data_ds = report_group['data']
        ranges = metadata['ranges']

        for cell in sim.net.compartCells:
            gid = int(cell.gid)
            if gid not in ranges:
                missing.append(gid)
                continue

            start, end = ranges[gid]
            im_report = np.asarray(data_ds[:, start:end], dtype=np.float32).T
            im_report = _reorder_im_report_if_needed(gid, im_report, start, end, metadata, state)

            expected_nseg = int(cell.getNumberOfSegments())
            if im_report.shape[0] != expected_nseg:
                raise ValueError(
                    f"SONATA i_membrane report segment count mismatch for gid {gid}: "
                    f"report has {im_report.shape[0]} segments, NetPyNE expects {expected_nseg}."
                )

            im_matrix = _copy_with_pad(im_report, nsteps)  # (nseg, nsteps)

            if sim.cfg.saveIMembrane and gid in sim.simData['iMembrane']:
                sim.simData['iMembrane'][gid][:, :] = im_matrix.T

            if sim.cfg.recordLFP:
                tr = sim.net.recXElectrode.getTransferResistance(gid)
                if tr.shape[1] != im_matrix.shape[0]:
                    raise ValueError(
                        f"Transfer-resistance shape mismatch for gid {gid}: "
                        f"tr has {tr.shape[1]} segments, i_membrane report has {im_matrix.shape[0]}."
                    )

                ecp = np.dot(tr, im_matrix)
                ecp_T = ecp.T.astype(np.float32)
                sim.simData['LFP'] += ecp_T

                if sim.cfg.saveLFPCells and gid in sim.simData['LFPCells']:
                    sim.simData['LFPCells'][gid][:, :] = ecp_T

                if sim.cfg.saveLFPPops and hasattr(sim.net, 'popForEachGid') and gid in sim.net.popForEachGid:
                    pop = sim.net.popForEachGid[gid]
                    if pop in sim.simData['LFPPops']:
                        sim.simData['LFPPops'][pop] += ecp_T

            if sim.cfg.recordDipole and hasattr(cell, 'M'):
                p = cell.M @ im_matrix
                p_T = p.T.astype(np.float64)

                sim.simData['dipoleSum'][:nsteps] += p_T

                if sim.cfg.saveDipoleCells and gid in sim.simData['dipoleCells']:
                    sim.simData['dipoleCells'][gid][:nsteps] = p_T

                if sim.cfg.saveDipolePops and hasattr(sim.net, 'popForEachGid') and gid in sim.net.popForEachGid:
                    pop = sim.net.popForEachGid[gid]
                    if pop in sim.simData['dipolePops']:
                        sim.simData['dipolePops'][pop][:nsteps] += p_T

            processed += 1

    state['processed_cells'] = processed
    state['missing_local_gids'] = missing

    if sim.rank == 0:
        print(
            '  Loaded SONATA i_membrane report '
            f"{metadata['population_path']} from {report_path} "
            f"(shape={metadata['data_shape']}, dt={time_info['dt']}, tstart={time_info['tstart']}, tstop={time_info['tstop']})."
        )
        if missing:
            print(f'  Warning: SONATA report did not contain {len(missing)} local gids on rank 0.')
