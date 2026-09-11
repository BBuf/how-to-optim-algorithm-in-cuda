#!/usr/bin/env bash
set -euo pipefail

sglang_repo="${1:?Usage: prepare_variants.sh /path/to/sglang /path/to/new-worktrees}"
variant_root="${2:?Provide a new output directory}"
commit=3b709e55c0f7599f90bdd400e1fe758c5a942cb6
small_batch=759baff47f2193008771f0886bb7947a1d91ac64
verify_moe=c36636b7da601639d4a8b19a5d008761252146a9

git -C "$sglang_repo" cat-file -e "$commit^{commit}"
mkdir -p "$variant_root"
variant_root="$(cd "$variant_root" && pwd)"
for variant in base middle final; do
  git -C "$sglang_repo" worktree add --detach "$variant_root/$variant" "$commit"
done
git -C "$variant_root/middle" revert --no-commit "$small_batch"
git -C "$variant_root/base" revert --no-commit "$small_batch"
git -C "$variant_root/base" revert --no-commit "$verify_moe"
for variant in base middle final; do
  git -C "$variant_root/$variant" diff HEAD --check
done
