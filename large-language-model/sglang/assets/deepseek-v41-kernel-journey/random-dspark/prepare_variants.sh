#!/usr/bin/env bash
set -euo pipefail

sglang_repo="${1:?Usage: prepare_variants.sh /path/to/sglang /path/to/new-worktrees}"
variant_root="${2:?Provide a new output directory}"
commit=835c39094ad017c2f54f8ea598002e669e6fa30d
small_batch_head=3b709e55c0f7599f90bdd400e1fe758c5a942cb6
index_projection_head=2e4dff1c4939c8589191e65340236b03b54dc84c
small_batch=759baff47f2193008771f0886bb7947a1d91ac64
verify_moe=c36636b7da601639d4a8b19a5d008761252146a9

git -C "$sglang_repo" cat-file -e "$commit^{commit}"
mkdir -p "$variant_root"
variant_root="$(cd "$variant_root" && pwd)"
for variant in base middle small_batch; do
  git -C "$sglang_repo" worktree add --detach "$variant_root/$variant" "$small_batch_head"
done
git -C "$sglang_repo" worktree add --detach "$variant_root/index_projection" "$index_projection_head"
git -C "$sglang_repo" worktree add --detach "$variant_root/final" "$commit"
git -C "$variant_root/middle" revert --no-commit "$small_batch"
git -C "$variant_root/base" revert --no-commit "$small_batch"
git -C "$variant_root/base" revert --no-commit "$verify_moe"
for variant in base middle small_batch index_projection final; do
  git -C "$variant_root/$variant" diff HEAD --check
done
