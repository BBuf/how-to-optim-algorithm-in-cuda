/******************************************************************************
 * Copyright (c) 2024 Colfax Research                                         *
 ******************************************************************************/
#pragma once
#include <string>

#include <cutlass/util/command_line.h>

struct Options {

  bool help;
  bool error;

  int m, n, k;
  float alpha, beta;
  unsigned seed;
  int timing_iterations;
  std::string task;
  int scheduler_num;

  Options() :
    help(false),
    error(false),
    m(8192),
    n(8192),
    k(8192),
    alpha(1.f),
    beta(0.f),
    seed(0u),
    timing_iterations(20),
    task("time"),
    scheduler_num(0)
  {}

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 8192);
    cmd.get_cmd_line_argument("n", n, 8192);
    cmd.get_cmd_line_argument("k", k, 8192);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("seed", seed, 0u);
    cmd.get_cmd_line_argument("iters", timing_iterations, 20);
    cmd.get_cmd_line_argument("task", task);
    cmd.get_cmd_line_argument("scheduler", scheduler_num, 0);
  }

  /// Prints the usage statement.
  std::ostream &print_usage(std::ostream &out) const {

    out << "Example demonstrating the same kernel with different tile schedulers.\n\n"
        << "Options:\n"
        << "  --help                      If specified, displays this usage "
           "statement\n\n"
        << "  --m=<int>                   M mode of problem size (only used with --single)\n"
        << "  --n=<int>                   N mode of problem size (only used with --single)\n"
        << "  --k=<int>                   K mode of problem size (constant across trials)\n"
        << "  --alpha=<f32>               Epilogue scalar alpha\n"
        << "  --beta=<f32>                Epilogue scalar beta\n\n"
        << "  --seed=<uint>               Set random seed (0: randomize from clock)\n"
        << "  --iters=<int>               Iterations (for time command)\n"
        << "  --task=[profile|time|validate|print_layouts|print_schedule|quantization_csv]\n"
        << "                              Task to perform (default: time)\n"
        << "  --scheduler=<int>           Number of scheduler to use\n"
        << "     (0: non-persistent, 1: data-parallel persistent, 2: stream-K, 3: heuristic)\n";

    return out;
  }
};
