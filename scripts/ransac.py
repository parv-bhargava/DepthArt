from scripts.utils import *


def import_into_colmap(
        path: Path,
        feature_dir: Path,
        database_path: str = "colmap.db",
) -> None:
    """Adds keypoints into colmap"""
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, path, "", "simple-radial", single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )
    db.commit()

