"""Command line: ``pynotation3d subplots``, ``project`` and ``backproject``."""
import argparse
from pathlib import Path

from .backproject import backproject_annotation
from .project import project_point_cloud
from .subplots import split_into_subplots, write_subplots


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pynotation3d")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("subplots", help="split a LAS into square subplots, each centred on (0, 0)")
    s.add_argument("las")
    s.add_argument("out_dir")
    s.add_argument("--size", type=float, nargs="+", default=[10.0], metavar="M",
                   help="tile side in metres, or X and Y sides (default: 10)")
    s.add_argument("--origin", type=float, nargs=2, metavar=("X", "Y"),
                   help="grid corner (default: the cloud's X/Y minimum)")
    s.add_argument("--min-points", type=int, default=1, help="leave out tiles with fewer points")

    p = sub.add_parser("project", help="classify with FuelDeep3D and write the four 2D projections")
    p.add_argument("las")
    p.add_argument("out_dir")
    p.add_argument("--method", choices=["full", "fast"], default="full",
                   help="full: TLS-SphereMap GPU features + projection; fast: CPU projection of the PNG channels only")
    p.add_argument("--fueldeep3d-dir", help="FuelDeep3D extdata/python directory")
    p.add_argument("--model", help="FuelDeep3D model checkpoint (default: bundled model)")
    p.add_argument("--res", type=float, default=0.2, help="angular resolution, degrees per pixel")
    p.add_argument("--recenter", action="store_true", help="project around the X/Y midpoint (off-centre crops)")
    p.add_argument("--overwrite", action="store_true", help="recompute existing intermediate outputs")

    b = sub.add_parser("backproject", help="map an annotated projection back onto the points")
    b.add_argument("annotation")
    b.add_argument("projection", help="*_projection.json or the project output directory")
    b.add_argument("label_map", help="dataset name in TLS-SphereMap's label_maps.json")
    b.add_argument("-o", "--out", help="output LAS")
    b.add_argument("--tolerance", type=int, default=3)
    b.add_argument("--drop-void", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "subplots":
        if len(args.size) not in (1, 2):
            parser.error("--size takes one or two values")
        size = args.size[0] if len(args.size) == 1 else tuple(args.size)
        subplots = split_into_subplots(args.las, size, origin=args.origin, min_points=args.min_points)
        paths = write_subplots(subplots, args.out_dir, Path(args.las).stem)
        print(f"Wrote {len(paths)} subplots to {args.out_dir}")
    elif args.command == "project":
        outputs = project_point_cloud(
            args.las, args.out_dir, method=args.method,
            fueldeep3d_dir=args.fueldeep3d_dir, model_path=args.model,
            h_res=args.res, v_res=args.res, recenter=args.recenter, overwrite=args.overwrite,
        )
        print(f"Wrote {outputs.metadata}")
    else:
        outputs = backproject_annotation(
            args.annotation, args.projection, args.label_map, args.out,
            tolerance=args.tolerance, drop_void=args.drop_void,
        )
        print(f"Wrote {outputs.las}; points per class: {outputs.class_counts}")


if __name__ == "__main__":
    main()
