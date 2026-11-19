import argparse

import yaml


def parse_ingestion_args(parser: argparse.ArgumentParser) -> None:
	"""
	Input: parser, the argument parser of main.py.
	Output: nothing.
	What it does: adds the ingestion arguments to parser.
	"""
	parser.add_argument(
		"--dataset_size",
		type=int,
		required=True,
		help="Number of documents to ingest")
	parser.add_argument(
		"--dataset_name",
		type=str,
		required=True,
		help="Name of the dataset, used as its output folder name")
	parser.add_argument(
		"--input_path",
		type=str,
		required=True,
		help="Local file or folder to ingest")
	parser.add_argument(
		"--output_path",
		type=str,
		required=True,
		help="Root output folder")
	parser.add_argument(
		"--document_type",
		type=str,
		required=True,
		help="Type of documents to keep, e.g. pdf")


def parse_args() -> argparse.Namespace:
	"""
	Input: none. Reads --config and any override flags from the command line.
	Output: an argparse.Namespace with one attribute per setting.
	What it does: reads the config file, turns each setting into its flag and parses them. A top-level key is one
	setting; a section such as ingestion holds settings. Flags given after --config override the file.
	"""
	# 1. Get the config file path. The other command line flags are overrides
	config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
	config_parser.add_argument("--config", type=str, required=True, help="YAML config file, e.g. config_temp.yaml")
	config_args, override_arguments = config_parser.parse_known_args()

	with open(config_args.config, encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)

	# 2. Turn each setting of the file into its flag
	config_arguments = []
	for key in config:
		value = config[key]
		if isinstance(value, dict):
			for setting_key in value:
				config_arguments.append("--" + setting_key)
				config_arguments.append(str(value[setting_key]))
		else:
			config_arguments.append("--" + key)
			config_arguments.append(str(value))

	# 3. Parse the file's flags and then the overrides, so an override replaces the file's value
	parser = argparse.ArgumentParser(description="RAG parameters", allow_abbrev=False)
	parser.add_argument(
		"--framework",
		type=str,
		required=True,
		help="rag or langchain")
	
	parse_ingestion_args(parser)

	parsed_args = parser.parse_args(config_arguments + override_arguments)
	return parsed_args
