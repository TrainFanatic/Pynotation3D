"""Command line: ``pynotation3d project`` and ``pynotation3d backproject``."""
import argparse

from .backproject import backproject_annotation
from .project import project_point_cloud


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pynotation3d")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("project", help="classify with FuelDeep3D and write the four 2D projections")
    p.add_argument("las")
    p.add_argument("out_dir")
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
    if args.command == "project":
        outputs = project_point_cloud(
            args.las, args.out_dir,
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
