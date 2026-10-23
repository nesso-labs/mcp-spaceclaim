import numpy as np
import pytest
import pyvista as pv

from mcp_spaceclaim import render

PNG = b"\x89PNG\r\n\x1a\n"


def cube_mesh(name="cube", size_m=0.02):
    poly = pv.Cube(x_length=size_m, y_length=size_m, z_length=size_m)
    return render.mesh_from_polydata(name, poly, scale=1000.0)


def test_mesh_is_scaled_to_working_units_and_triangulated():
    mesh = cube_mesh()
    assert mesh.triangles.shape == (12, 3)
    assert np.allclose(mesh.points.max(axis=0), [10, 10, 10])


@pytest.mark.parametrize("views", [["iso"], ["front", "top", "right", "iso"], ["top", "bottom"]])
def test_render_produces_a_png(views):
    scene = render.Scene(meshes=[cube_mesh(), cube_mesh("other", 0.01)], unit="mm")
    png = render.render(scene, views, size=400)
    assert png.startswith(PNG)
    assert len(png) > 5_000


def test_highlights_render():
    face = cube_mesh("face")
    face.highlight = True
    scene = render.Scene(
        meshes=[cube_mesh(), face], lines=[np.array([[0, 0, 0], [10, 10, 10]])], unit="mm"
    )
    assert render.render(scene, ["iso"], size=300).startswith(PNG)


def test_render_errors():
    with pytest.raises(ValueError, match="no bodies"):
        render.render(render.Scene(), ["iso"])
    with pytest.raises(ValueError, match="unknown view"):
        render.render(render.Scene(meshes=[cube_mesh()]), ["sideways"])
