"""Add models from Hugging Face: search, inspect a repo, add it to the Library as a recipe."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import hf, recipes

router = APIRouter(prefix="/api/hf", tags=["huggingface"])


class RepoBody(BaseModel):
    repo: str


@router.get("/search")
def search(q: str = "", kind: str = "all") -> dict:
    if kind not in ("all", "image", "video"):
        raise HTTPException(400, "kind must be all, image or video")
    try:
        return {"results": hf.search(q, kind)}
    except hf.HFError as exc:
        raise HTTPException(502, str(exc))


def _inspect(repo: str) -> dict:
    try:
        result = hf.inspect(repo)
    except hf.HFError as exc:
        raise HTTPException(502, str(exc))
    existing = recipes.find_by_repo(result["repo"])
    result["already_in_library"] = existing["id"] if existing else None
    result["existing_name"] = existing["name"] if existing else None
    return result


@router.post("/inspect")
def inspect(body: RepoBody) -> dict:
    return _inspect(body.repo)


@router.post("/add")
def add(body: RepoBody) -> dict:
    result = _inspect(body.repo)
    if result["already_in_library"]:
        raise HTTPException(409, f"{result['existing_name']} is already in your Library.")
    if result["unsupported"] or not result["recipe"]:
        raise HTTPException(422, result["unsupported"] or "This model can't be added.")
    recipe = result["recipe"]
    if recipes.get(recipe["id"]):
        raise HTTPException(409, f"A model with the id '{recipe['id']}' already exists.")
    try:
        recipes.save_user(recipe)
    except (ValueError, OSError) as exc:
        raise HTTPException(500, f"Could not save the recipe: {exc}")
    return {"ok": True, "id": recipe["id"], "name": recipe["name"]}
