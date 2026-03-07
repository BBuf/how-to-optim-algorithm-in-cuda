.PHONY: update style style-unsafe quality cstyle cquality clist

root_dir := .

CLANG_FORMAT_EXCLUDE_DIRS := build third-party external tests .git .vscode
CLANG_FORMAT_EXTS := c cpp h hpp cu cuh cc cxx
CLANG_FORMAT_EXCLUDE_ARGS := $(foreach dir,$(CLANG_FORMAT_EXCLUDE_DIRS),-not -path "./$(dir)/*")
CLANG_FORMAT_EXT_ARGS := $(firstword $(CLANG_FORMAT_EXTS))
CLANG_FORMAT_EXT_ARGS := -name "*.$(CLANG_FORMAT_EXT_ARGS)"
CLANG_FORMAT_EXT_ARGS += $(foreach ext,$(wordlist 2,$(words $(CLANG_FORMAT_EXTS)),$(CLANG_FORMAT_EXTS)),-o -name "*.$(ext)")

update:
	git submodule update --init --remote --recursive

style:
	ruff check $(root_dir) --fix
	ruff format $(root_dir)

style-unsafe:
	ruff check $(root_dir) --fix --unsafe-fixes
	ruff format $(root_dir)

quality:
	ruff check $(root_dir)
	ruff format --check $(root_dir)

cstyle:
	find $(root_dir) -type f \( $(CLANG_FORMAT_EXT_ARGS) \) $(CLANG_FORMAT_EXCLUDE_ARGS) | xargs clang-format -i

cquality:
	find $(root_dir) -type f \( $(CLANG_FORMAT_EXT_ARGS) \) $(CLANG_FORMAT_EXCLUDE_ARGS) | xargs clang-format -n -Werror

clist:
	find $(root_dir) -type f \( $(CLANG_FORMAT_EXT_ARGS) \) $(CLANG_FORMAT_EXCLUDE_ARGS)
