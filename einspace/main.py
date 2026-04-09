"""
Drop-in replacement for einspace/main.py that adds a new search strategy:

    search_strategy: tree_bo

Everything else in the repo remains unchanged. The only new logic is the branch
that instantiates TreeKernelBO.
"""

from argparse import ArgumentParser
from itertools import repeat
from pprint import pprint

import torch
import yaml
from foresight.pruners.predictive import find_measures

from einspace.compiler import Compiler
from einspace.data import get_data_loaders
from einspace.network import Network
from einspace.search_spaces import EinSpace
from einspace.search_strategies import RandomSearch, RegularisedEvolution
from einspace.seed_architectures import seed_architectures
from einspace.trainers import Trainer
from einspace.tree_kernel_bo import TreeKernelBO
from einspace.utils import get_exp_name, set_seed


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Config loading.
    # ---------------------------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Convert YAML string "None" into Python None.
    for key, value in config.items():
        if value == "None":
            config[key] = None

    if args.device is not None:
        config["device"] = args.device
    else:
        raise ValueError("Please specify device")

    if "train_mode" not in config:
        config["train_mode"] = ["train", "val"]

    pprint(config)
    set_seed(config["seed"])
    device = config["device"] if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------------
    # Dataset / dataloaders.
    # ---------------------------------------------------------------------
    (
        original_train_loader,
        original_val_loader,
        original_trainval_loader,
        original_test_loader,
    ) = get_data_loaders(
        dataset=config["dataset"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        load_in_gpu=config["load_in_gpu"],
        device=device,
    )

    if config["train_mode"][0] == "train":
        train_loader = original_train_loader
    elif config["train_mode"][0] == "trainval":
        train_loader = original_trainval_loader
    else:
        raise ValueError(f"Unknown train split mode: {config['train_mode'][0]}")

    if config["train_mode"][1] == "val":
        val_loader = original_val_loader
    elif config["train_mode"][1] == "test":
        val_loader = original_test_loader
    else:
        raise ValueError(f"Unknown eval split mode: {config['train_mode'][1]}")

    test_loader = original_test_loader

    # ---------------------------------------------------------------------
    # Search space + compiler.
    # ---------------------------------------------------------------------
    einspace = EinSpace(
        input_shape=(
            config["batch_size"],
            config["channels"],
            *config["image_size"],
        ),
        input_mode=config["input_mode"],
        num_repeated_cells=config["search_space_num_repeated_cells"],
        device=device,
        computation_module_prob=config["search_space_computation_module_prob"],
        min_module_depth=config["search_space_min_module_depth"],
        max_module_depth=config["search_space_max_module_depth"],
    )
    compiler = Compiler()

    def evaluation_fn(architecture, modules):
        model = Network(
            modules,
            architecture["output_shape"],
            config["num_classes"],
            config,
        )
        trainer = Trainer(
            model,
            device=device,
            train_dataloader=train_loader,
            valid_dataloader=val_loader,
            test_dataloader=None,
            config=config,
            logger=None,
        )
        best = trainer.train()
        return best

    exp_name = get_exp_name(**config)
    print(exp_name)

    # ---------------------------------------------------------------------
    # Existing search strategies.
    # ---------------------------------------------------------------------
    if config["search_strategy"] in ["re", "rm"]:
        if (
            config["search_strategy_architecture_seed"] is not None
            and "warmup" in config["search_strategy_architecture_seed"]
        ):

            def zero_cost_evaluation_fn(architecture, modules):
                model = Network(
                    modules,
                    architecture["output_shape"],
                    config["num_classes"],
                    config,
                )
                zero_cost_proxy = config["search_strategy_architecture_seed"].split("_")[-1]
                measure = find_measures(
                    model,
                    train_loader,
                    ("random", 1, config["num_classes"]),
                    device=device,
                    measure_names=[zero_cost_proxy],
                )
                return {"val_score": measure}

            random_search = RandomSearch(
                search_space=einspace,
                compiler=compiler,
                evaluation_fn=zero_cost_evaluation_fn,
                num_samples=config["search_strategy_warmup_samples"],
                save_name=exp_name,
                continue_search=config["search_strategy_continue_search"],
            )
            warmup_history = random_search.search()
            architecture_seed = warmup_history[: config["search_strategy_init_pop_size"]]
        elif (
            config["search_strategy_architecture_seed"] is not None
            and "+" in config["search_strategy_architecture_seed"]
        ):
            seed_mix = config["search_strategy_architecture_seed"].split("+")
            k = config["search_strategy_init_pop_size"] // len(seed_mix)
            architecture_seed = [
                seed_architectures[arch] for arch in repeat(seed_mix, k)
            ]
        elif config["search_strategy_architecture_seed"] is not None:
            architecture_seed = [
                seed_architectures[config["search_strategy_architecture_seed"]]
            ]
        else:
            architecture_seed = []

        search_strategy = RegularisedEvolution(
            search_space=einspace,
            compiler=compiler,
            evaluation_fn=evaluation_fn,
            num_samples=config["search_strategy_num_samples"],
            init_pop_size=config["search_strategy_init_pop_size"],
            sample_size=config["search_strategy_sample_size"],
            save_name=exp_name,
            continue_search=config["search_strategy_continue_search"],
            architecture_seed=architecture_seed,
            update_population=config["search_strategy"] == "re",
        )

    elif config["search_strategy"] == "rs":
        search_strategy = RandomSearch(
            search_space=einspace,
            compiler=compiler,
            evaluation_fn=evaluation_fn,
            num_samples=config["search_strategy_num_samples"],
            save_name=exp_name,
            continue_search=config["search_strategy_continue_search"],
        )

    # ---------------------------------------------------------------------
    # NEW: Tree-kernel Bayesian optimization.
    # ---------------------------------------------------------------------
    elif config["search_strategy"] == "tree_bo":
        if config.get("search_strategy_architecture_seed") is not None:
            if "+" in config["search_strategy_architecture_seed"]:
                seed_mix = config["search_strategy_architecture_seed"].split("+")
                architecture_seed = [seed_architectures[arch] for arch in seed_mix]
            else:
                architecture_seed = [
                    seed_architectures[config["search_strategy_architecture_seed"]]
                ]
        else:
            architecture_seed = []

        search_strategy = TreeKernelBO(
            search_space=einspace,
            compiler=compiler,
            evaluation_fn=evaluation_fn,
            num_samples=config["search_strategy_num_samples"],
            init_num_samples=config["tree_bo_init_num_samples"],
            candidate_pool_size=config["tree_bo_candidate_pool_size"],
            acquisition=config["tree_bo_acquisition"],
            save_name=exp_name,
            continue_search=config["search_strategy_continue_search"],
            mutation_prob=config["tree_bo_mutation_prob"],
            local_tournament_size=config["tree_bo_local_tournament_size"],
            gp_noise=config["tree_bo_gp_noise"],
            acq_xi=config["tree_bo_ei_xi"],
            acq_beta=config["tree_bo_ucb_beta"],
            architecture_seed=architecture_seed,
        )

    else:
        raise ValueError(f"Unknown search strategy: {config['search_strategy']}")

    # Run search.
    history = search_strategy.search()

    # ---------------------------------------------------------------------
    # Retrain the best discovered architecture on train+val and test.
    # This matches the original main.py flow.
    # ---------------------------------------------------------------------
    best_individual = max(history, key=lambda individual: individual.accuracy)

    best_model = Network(
        compiler.compile(best_individual.arch),
        best_individual.arch["output_shape"],
        config["num_classes"],
        config,
    )

    if isinstance(best_individual.hpo_dict, dict) and len(best_individual.hpo_dict) > 0:
        config["lr"] = best_individual.hpo_dict["lr"]
        config["momentum"] = best_individual.hpo_dict["momentum"]
        config["weight_decay"] = best_individual.hpo_dict["weight_decay"]

    config["epochs"] = config["test_epochs"]

    trainer = Trainer(
        best_model,
        device=device,
        train_dataloader=original_trainval_loader,
        valid_dataloader=None,
        test_dataloader=original_test_loader,
        config=config,
        logger=None,
    )
    best = trainer.train()
    print(f"Best accuracy: {best.accuracy} at epoch {best.epoch}")
