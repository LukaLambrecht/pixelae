#!/usr/bin/env python3
"""
Summarize flagged-lumisection anomalies from one or more
nmf_testing_pattern.py output files (flagged_lumisections_*.json).

Reads the multi-layer flags (flagged_run_numbers/flagged_ls_numbers) and,
if present, the single-layer informational flags
(flagged_run_numbers_sl/flagged_ls_numbers_sl) from each input file, takes
the union across all inputs (so passing the same lumisection in more than
one file, e.g. a whole-era file and its own part-split files, does not
double-count it), and groups them into "anomalies" per the rule below.
Writes one CSV row per anomaly.

Anomaly-counting rule
----------------------
If the input file(s) have the flagged_topology/flagged_topology_sl key
(spatial bounding boxes per contributing layer, per flagged LS), anomalies
are tracked spatially: a new anomaly starts whenever a loss region appears
that does not spatially overlap any currently-open anomaly, and it keeps
running for as long as that region keeps reappearing in roughly the same
place -- bridging gaps of less than --threshold LS in between (e.g. an
unrelated filter briefly excluding a lumisection from evaluation), same as
before, but now gated on spatial overlap too so that temporally-adjacent
but spatially-distinct anomalies (different layers, or different blobs
within the same layer) are correctly counted as separate anomalies instead
of being merged into one just because they happen to border each other in
time. See topology_anomalies_for().

If topology info is not available (input file(s) predate that feature),
falls back to the plain LS-gap rule: consecutive flagged lumisections in
the same run are one anomaly, and two ranges separated by a gap of less
than --threshold lumisections are merged into one -- i.e. --threshold is
the smallest gap (in LS) that will NOT be merged. See anomalies_for().

Output columns
---------------
run, first lumisection, last lumisection, number of lumisections (the
actually-flagged count, not the span), layers (';'-joined list of the
genuinely contributing layers, union across the anomaly's flagged LS --
empty if the input files predate the topology feature), type.

Printout
---------
When more than one input file is given, prints a per-file breakdown (each
file summarized independently) above the aggregated (union-across-files)
result, purely informational -- the output CSV is always the aggregated
result only, never per-file. --verbose additionally prints the exact
contents of the output CSV to stdout right before it's written, so it can
be checked directly in the printout too. --skip-single-layer skips
reading and writing single-layer anomalies entirely (only multi-layer
flags are read from the input files, reported, and written).

Note: --threshold was only calibrated/validated against the DP note's
multi-layer anomaly counts. Single-layer flags are much denser per run, so
the same threshold tends to over-merge them into long combined anomalies --
use --threshold-sl to set a different (e.g. 0, for no merging) value for
the single-layer category specifically.

Layer filtering
-----------------
--include-layers/--exclude-layers keep or drop whole anomalies (both
multi- and single-layer) based on which layers genuinely contributed to
them: --include-layers keeps an anomaly only if ALL its layers are in the
given list (default: no filtering, i.e. all layers allowed); --exclude-
layers keeps an anomaly only if NONE of its layers are in the given list
(default: none excluded). Requires layer info (i.e. the topology feature)
to be meaningful -- an anomaly with no layer info at all (pre-topology
input files) is always kept, since it can't be evaluated against either
filter. Needs topology info to make sense: it operates on the "layers"
already derived per anomaly (see the anomaly-counting rule above), not on
a raw per-LS layer filter.

Usage:
  python summarize_anomalies.py file1.json [file2.json ...] [-o anomalies.csv] [--threshold 9] [--threshold-sl 0] [--exclude-runs 381594]
"""
import argparse
import csv
import json
import os


def load_pairs(files, include_sl=True):
    """Returns (ml, sl, ml_layers, sl_layers, ml_topology, sl_topology, has_ml_topology, has_sl_topology):
    - ml, sl: sets of (run, ls) pairs for multi-layer / single-layer flags.
    - ml_layers, sl_layers: dict (run, ls) -> set of genuinely (non-masked)
      contributing layer names, derived from flagged_topology/flagged_topology_sl
      if present in the input file, empty otherwise for backwards compatibility
      with older output files that predate the topology feature.
    - ml_topology, sl_topology: dict (run, ls) -> list of {'layer', 'bbox'}
      for the genuinely (non-masked) contributing blobs, used for the
      spatial track-based anomaly grouping.
    - has_ml_topology, has_sl_topology: whether any input file provided the
      flagged_topology / flagged_topology_sl key at all (regardless of
      whether any given flagged LS ended up with a non-empty blob list) --
      used to decide whether to use the spatial grouping or fall back to
      the plain LS-gap rule for files that predate this feature.
    If include_sl is False, the single-layer fields are not even read from
    the input files, and sl/sl_layers/sl_topology/has_sl_topology come back
    empty/False."""
    ml, sl = set(), set()
    ml_layers, sl_layers = {}, {}
    ml_topology, sl_topology = {}, {}
    has_ml_topology, has_sl_topology = False, False
    for path in files:
        with open(path) as f:
            d = json.load(f)
        ml_pairs = list(zip(d['flagged_run_numbers'], d['flagged_ls_numbers']))
        ml |= set(ml_pairs)
        if 'flagged_topology' in d: has_ml_topology = True
        for pair, blobs in zip(ml_pairs, d.get('flagged_topology', [[]] * len(ml_pairs))):
            keep = [{'layer': b['layer'], 'bbox': b['bbox']} for b in blobs if not b['masked']]
            ml_layers.setdefault(pair, set()).update(b['layer'] for b in keep)
            ml_topology.setdefault(pair, []).extend(keep)
        if not include_sl:
            continue
        sl_pairs = list(zip(d.get('flagged_run_numbers_sl', []), d.get('flagged_ls_numbers_sl', [])))
        sl |= set(sl_pairs)
        if 'flagged_topology_sl' in d: has_sl_topology = True
        for pair, blobs in zip(sl_pairs, d.get('flagged_topology_sl', [[]] * len(sl_pairs))):
            keep = [{'layer': b['layer'], 'bbox': b['bbox']} for b in blobs if not b['masked']]
            sl_layers.setdefault(pair, set()).update(b['layer'] for b in keep)
            sl_topology.setdefault(pair, []).extend(keep)
    return ml, sl, ml_layers, sl_layers, ml_topology, sl_topology, has_ml_topology, has_sl_topology


def strict_ranges(pairs):
    """Group (run, ls) pairs into maximal strictly-consecutive-LS ranges per run."""
    ranges = []
    for run, ls in sorted(pairs):
        if ranges and ranges[-1][0] == run and ls == ranges[-1][2] + 1:
            ranges[-1][2] = ls
        else:
            ranges.append([run, ls, ls])
    return ranges


def merge_by_gap(ranges, threshold):
    """Merge same-run ranges separated by a gap of less than `threshold` LS
    -- i.e. `threshold` is the smallest gap that will NOT be merged."""
    merged = []
    for r in sorted(ranges):
        if merged and merged[-1][0] == r[0] and (r[1] - merged[-1][2] - 1) < threshold:
            merged[-1][2] = r[2]
        else:
            merged.append(list(r))
    return merged


def anomalies_for(pairs, threshold, exclude_runs, layers_by_pair=None):
    """Returns a list of (run, first, last, n_flagged, layers) per anomaly, where
    n_flagged is the number of *actually flagged* LS within [first, last]
    -- not the span (last - first + 1), since a merged anomaly can bridge
    a gap of up to `threshold` LS that were never flagged (e.g. excluded
    by a filter, or just not caught by the pattern-matching) -- and layers
    is the sorted list of layers (union across the anomaly's flagged LS)
    that genuinely contributed, or an empty list if no layer info is
    available (older input files that predate the layer-info feature).
    Purely LS-gap based (no spatial awareness) -- see topology_anomalies_for
    for the spatially-aware version used when topology info is available."""
    pairs = {(r, l) for r, l in pairs if r not in exclude_runs}
    flagged_by_run = {}
    for r, l in pairs:
        flagged_by_run.setdefault(r, set()).add(l)
    merged = merge_by_gap(strict_ranges(pairs), threshold)
    anomalies = []
    for run, first, last in merged:
        flagged_ls = [ls for ls in range(first, last + 1) if ls in flagged_by_run[run]]
        layers = set()
        if layers_by_pair is not None:
            for ls in flagged_ls:
                layers.update(layers_by_pair.get((run, ls), ()))
        anomalies.append((run, first, last, len(flagged_ls), sorted(layers)))
    return anomalies


def bbox_overlap(a, b):
    """True if two [rmin, rmax, cmin, cmax] boxes intersect (inclusive
    bounds), or either is None -- a "no spatial info for this LS" wildcard
    that matches whatever track is temporally closest, used to gracefully
    handle a flagged LS without any non-masked blob (rare edge case) mixed
    in among LS that do have topology info."""
    if a is None or b is None:
        return True
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def bbox_union(a, b):
    if a is None: return b
    if b is None: return a
    return [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])]


def topology_anomalies_for(pairs, topology_by_pair, threshold, exclude_runs):
    """Spatially-aware version of anomalies_for(): groups (run, ls) pairs
    into anomalies ("tracks") using the per-LS spatial blobs in
    topology_by_pair ((run, ls) -> list of {'layer', 'bbox'}) instead of
    pure LS-gap merging. For each flagged LS, in run order, every
    contributing blob is matched against the currently-open tracks in that
    run: a track matches if the gap since it was last updated is less than
    `threshold` LS (i.e. `threshold` is the smallest gap that will NOT be
    bridged) AND its (union) bounding box spatially overlaps the blob's; if none match,
    a new track starts. A track's bbox grows (via union) as it absorbs
    matching blobs over time, and its layer set accumulates every layer
    that ever contributed to it. This way a new loss region starts a new
    anomaly immediately, even if it appears right next to (in time) an
    unrelated, already-open one elsewhere, while the same physical region
    reappearing after a short gap still counts as one continuous anomaly.
    Returns a list of (run, first, last, n_flagged, layers) per anomaly,
    with the same meaning as anomalies_for()."""
    pairs = sorted({(r, l) for r, l in pairs if r not in exclude_runs})
    anomalies = []
    runs = sorted(set(r for r, _ in pairs))
    for run in runs:
        lss = sorted(l for r, l in pairs if r == run)
        tracks = []  # each: {'bbox', 'first', 'last', 'ls_set', 'layers'}
        for ls in lss:
            blobs = topology_by_pair.get((run, ls)) or [{'layer': None, 'bbox': None}]
            for blob in blobs:
                match = None
                for tr in tracks:
                    if (ls - tr['last'] - 1) >= threshold:
                        continue
                    if bbox_overlap(blob['bbox'], tr['bbox']):
                        match = tr
                        break
                if match is None:
                    match = {'bbox': None, 'first': ls, 'last': ls, 'ls_set': set(), 'layers': set()}
                    tracks.append(match)
                match['last'] = ls
                match['ls_set'].add(ls)
                match['bbox'] = bbox_union(match['bbox'], blob['bbox'])
                if blob['layer'] is not None:
                    match['layers'].add(blob['layer'])
        for tr in tracks:
            anomalies.append((run, tr['first'], tr['last'], len(tr['ls_set']), sorted(tr['layers'])))
    return anomalies


def compute_anomalies(pairs, topology_by_pair, threshold, exclude_runs, has_topology, layers_by_pair):
    """Dispatches to topology_anomalies_for() if topology info is available,
    else falls back to the plain LS-gap anomalies_for() -- shared by both
    the per-file and the aggregated-across-files reporting in main()."""
    if has_topology:
        return topology_anomalies_for(pairs, topology_by_pair, threshold, exclude_runs)
    return anomalies_for(pairs, threshold, exclude_runs, layers_by_pair)


def filter_by_layers(anomalies, include_layers, exclude_layers):
    """Keeps only anomalies whose contributing layers pass both filters:
    - include_layers (default None, meaning "all layers included"): the
      anomaly's layers must *all* be in this list.
    - exclude_layers (default none): the anomaly's layers must *none* be
      in this list.
    An anomaly with no layer info at all (pre-topology input files, where
    "layers" is always empty) is not filterable and is always kept."""
    kept = []
    for a in anomalies:
        layers = set(a[4])
        if layers:
            if include_layers is not None and not layers.issubset(include_layers):
                continue
            if exclude_layers and layers & exclude_layers:
                continue
        kept.append(a)
    return kept


def report(label, n_flagged, anomalies):
    n_anom = len(anomalies)
    avg_dur = sum(a[3] for a in anomalies) / n_anom if n_anom else 0
    print(f'{label}: {n_flagged} flagged LS, {n_anom} anomalies, avg duration {avg_dur:.2f} LS (of actually-flagged LS)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+', help='One or more flagged_lumisections_*.json files.')
    parser.add_argument('-o', '--output', default='anomalies.csv', help='Output CSV path (default: anomalies.csv).')
    parser.add_argument('--threshold', type=int, default=10,
                         help='Smallest gap (in LS) that will NOT be merged -- same-run, same-location multi-layer '
                              'anomalies separated by less than this are merged into one '
                              '(default: 10; calibrated against the DP note\'s multi-layer anomaly counts, '
                              'see the module docstring -- not validated for single-layer, see --threshold-sl).')
    parser.add_argument('--threshold-sl', type=int, default=None,
                         help='Same as --threshold, but for single-layer anomalies specifically. '
                              'Defaults to --threshold if not given. Single-layer flags are much denser '
                              'per run than multi-layer ones, so the same threshold tends to over-merge '
                              'them into long combined anomalies; pass e.g. --threshold-sl 0 for genuinely '
                              'unmerged single-layer episodes.')
    parser.add_argument('--exclude-runs', type=int, nargs='*', default=[],
                         help='Run numbers to exclude entirely (e.g. known instrumental artifacts).')
    parser.add_argument('--include-layers', nargs='*', default=None,
                         help='Only count an anomaly (multi- or single-layer) if ALL of its contributing '
                              'layers are in this list (default: all layers included, i.e. no filtering). '
                              'An anomaly with no layer info at all -- pre-topology input files -- is always kept.')
    parser.add_argument('--exclude-layers', nargs='*', default=[],
                         help='Only count an anomaly (multi- or single-layer) if NONE of its contributing '
                              'layers are in this list (default: none excluded). Applied after --include-layers. '
                              'An anomaly with no layer info at all -- pre-topology input files -- is always kept.')
    parser.add_argument('--skip-single-layer', action='store_true',
                         help='Do not read or write single-layer (informational) anomalies at all -- only '
                              'multi-layer flags are read from the input files, reported, and written to the '
                              'output CSV.')
    parser.add_argument('-v', '--verbose', action='store_true',
                         help='Print the full contents of the output CSV to stdout just before writing it.')
    args = parser.parse_args()
    if args.threshold_sl is None:
        args.threshold_sl = args.threshold
    include_layers = set(args.include_layers) if args.include_layers is not None else None
    exclude_layers = set(args.exclude_layers)

    exclude = set(args.exclude_runs)

    layer_filter_active = include_layers is not None or exclude_layers

    def flagged_count(pairs, anomalies):
        # with a layer filter active, the raw unique-pairs count no longer
        # matches "what's being reported" (some pairs belong only to
        # filtered-out anomalies), so fall back to summing the surviving
        # anomalies' counts instead; without a filter, keep the original,
        # merge-independent definition (needed for cross-checks against
        # e.g. the production JSON's raw flagged-LS totals).
        if layer_filter_active:
            return sum(a[3] for a in anomalies)
        return len({(r, l) for r, l in pairs if r not in exclude})

    include_sl = not args.skip_single_layer

    if len(args.inputs) > 1:
        print('Per-file breakdown (each file summarized on its own, before taking the union across all inputs):')
        for path in args.inputs:
            (fml_pairs, fsl_pairs, fml_layers, fsl_layers,
             fml_topology, fsl_topology, fhas_ml_topology, fhas_sl_topology) = load_pairs([path], include_sl=include_sl)
            fml_anomalies = filter_by_layers(
                compute_anomalies(fml_pairs, fml_topology, args.threshold, exclude, fhas_ml_topology, fml_layers),
                include_layers, exclude_layers)
            n_fml_flagged = flagged_count(fml_pairs, fml_anomalies)
            has_topology = fhas_ml_topology and (fhas_sl_topology if include_sl else True)
            tag = '' if has_topology else ' (no topology info -- plain LS-gap grouping)'
            print(f'  {os.path.basename(path)}{tag}')
            report('    Multi-layer', n_fml_flagged, fml_anomalies)
            if include_sl:
                fsl_anomalies = filter_by_layers(
                    compute_anomalies(fsl_pairs, fsl_topology, args.threshold_sl, exclude, fhas_sl_topology, fsl_layers),
                    include_layers, exclude_layers)
                n_fsl_flagged = flagged_count(fsl_pairs, fsl_anomalies)
                report('    Single-layer', n_fsl_flagged, fsl_anomalies)
        print()

    (ml_pairs, sl_pairs, ml_layers, sl_layers,
     ml_topology, sl_topology, has_ml_topology, has_sl_topology) = load_pairs(args.inputs, include_sl=include_sl)

    if not has_ml_topology:
        print('WARNING: no flagged_topology found in the input file(s) -- falling back to plain '
              'LS-gap grouping (no spatial awareness) for multi-layer anomalies.')
    if include_sl and not has_sl_topology:
        print('WARNING: no flagged_topology_sl found in the input file(s) -- falling back to plain '
              'LS-gap grouping (no spatial awareness) for single-layer anomalies.')
    ml_anomalies = filter_by_layers(
        compute_anomalies(ml_pairs, ml_topology, args.threshold, exclude, has_ml_topology, ml_layers),
        include_layers, exclude_layers)
    n_ml_flagged = flagged_count(ml_pairs, ml_anomalies)

    print(f'Read {len(args.inputs)} input file(s).')
    print('Aggregated results (union across all input files):')
    report('Multi-layer', n_ml_flagged, ml_anomalies)

    if include_sl:
        sl_anomalies = filter_by_layers(
            compute_anomalies(sl_pairs, sl_topology, args.threshold_sl, exclude, has_sl_topology, sl_layers),
            include_layers, exclude_layers)
        n_sl_flagged = flagged_count(sl_pairs, sl_anomalies)
        report('Single-layer', n_sl_flagged, sl_anomalies)
    else:
        sl_anomalies = []

    rows = [['run', 'first lumisection', 'last lumisection', 'number of lumisections', 'layers', 'type']]
    for run, first, last, n_flagged, layers in ml_anomalies:
        rows.append([run, first, last, n_flagged, ';'.join(layers), 'multi-layer'])
    for run, first, last, n_flagged, layers in sl_anomalies:
        rows.append([run, first, last, n_flagged, ';'.join(layers), 'single-layer'])

    if args.verbose:
        print(f'Contents of {args.output} (about to be written):')
        for row in rows:
            print('  ' + ','.join(str(x) for x in row))

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f'Wrote {len(rows) - 1} anomalies to {args.output}')


if __name__ == '__main__':
    main()
