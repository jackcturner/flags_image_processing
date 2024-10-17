#!/usr/bin/env Rscript

# Import the relevent libraries.
library(ProFound)
library(Rfits)
library(Rwcs)
library(hash)
library(yaml)
library(glue)
library(rhdf5)
library(stringr)

# Function for identifying and replacing environment variables.
expand_env_var <- function(string) {
  pattern <- "\\$\\{?([A-Za-z_][A-Za-z0-9_]*)\\}?"

  str_replace_all(string, pattern, function(match) {
    var_name <- gsub("\\$\\{?([A-Za-z_][A-Za-z0-9_]*)\\}?", "\\1", match)
    Sys.getenv(var_name, unset = "")
  })
}

# Function to parse command-line arguments.
parse_args <- function(args) {

  # Initialize an empty list to store the arguments
  arg_list <- list()

  # Loop through each argument
  for (arg in args) {
    # Split the argument on '='
    parts <- strsplit(arg, "=")[[1]]

    # Ensure we have exactly two parts.
    if (length(parts) == 2) {
      arg_name <- parts[1]
      arg_value <- parts[2]

      # Try to convert the value to numeric if possible.
      numeric_value <- suppressWarnings(as.numeric(arg_value))

      # Store the argument in the list, convert to numeric if it's a valid number.
      if (!is.na(numeric_value)) {
        arg_list[[arg_name]] <- numeric_value
      } else {
        arg_list[[arg_name]] <- arg_value
      }
    } else {
      warning(paste("Invalid argument format:", arg))
    }
  }
  return(arg_list)
}

# This is the main ProFound class.
# Written to be useable directly from R if required.
ProFound <- setRefClass("profound", fields = list(config_path = "character", config = "list"),

                        methods = list(initialize = function(config_path) {
                          # Store the config filepath.
                          .self$config_path <- config_path
                          # Store config file content.
                          .self$config <- yaml.load_file(config_path)
                        },

                        extract = function(science, parameters = hash(), outputs = NULL, outdir = NULL) {

                          # If no output directory given, use the current directory.
                          if (is.null(outdir) == TRUE) {
                            outdir <- getwd()
                          }

                          # Make a local copy of the config.
                          config <- .self$config

                          # Update the config with given parameters.
                          if (length(parameters) > 0) {
                            for (key in names(parameters)) {
                              config[[key]] <- parameters[[key]]
                            }
                          }
                          # Store the config as is for saving as hdf5 attributes.
                          att_config <- config

                          for (key in names(config)) {
                            if (typeof(config[[key]]) == "character") {
                              # Convert string Infs to objects.
                              if (config[[key]] == "Inf") {
                                config[[key]] <- Inf
                                # Convert "None" to NULL
                              } else if (config[[key]] == "None") {
                                config[[key]] <- NULL
                                # Expand environment variables.
                              } else {
                                config[[key]] <- expand_env_var(config[[key]])
                              }
                            }
                          }

                          # Read in precomputed segmentation image,
                          if (is.null(config[["segim"]]) == FALSE) {
                            segim <- Rfits_read_image(config[["segim"]])
                            config[["segim"]] <- segim$imDat
                            rm(segim)
                          }
                          # and the mask file if required.
                          if (typeof(config[["mask"]]) == "character") {
                            mask <- Rfits_read_image(config[["mask"]])
                            config[["mask"]] <- mask$imDat
                            rm(mask)
                          }

                          # Read in the PSF file if provided.
                          if (is.null(config[["psf"]]) == FALSE) {
                            psf <- Rfits_read_image(config[["psf"]])
                            config[["psf"]] <- psf$imDat
                            rm(psf)
                          }

                          # If only one image is given, use it as detection and measurement.
                          if (length(science) == 1) {
                            print("Running in single image mode.")
                            sci <- Rfits_read_image(science)
                            input <- list(sci, sci)
                            cat_name <- glue("{outdir}/{gsub('.fits', '_profound.hdf5', basename(science))}")

                            # If two are given, the first is the detection image, second is the measurement.
                          } else if (length(science) == 2) {
                            print("Running in double image mode.")
                            det <- Rfits_read_image(unlist(science[1]))
                            sci <- Rfits_read_image(unlist(science[2]))
                            input <- list(det, sci)
                            cat_name <- glue("{outdir}/{gsub('.fits', '_profound.hdf5', basename(unlist(science[2])))}")
                            rm(det)

                            # Else, raise an error.
                          } else {
                            stop("science is not in an acceptable format. Should either be a path to a measurement image 
                            or a list containing the paths to a detection and measurement image.")
                          }
                          hdr <- sci$keyvalues
                          rm(sci)

                          # Some parameters need to be set manually.
                          manual <- list(inputlist = input, detectbands = "det", multibands = c("det", "sci"), keepim = FALSE, fluxtype = "Jansky",
                                         totappend = "_total", colappend = "_colour", grpappend = "_group")
                          all_params <- c(config, manual)

                          # Run ProFound.
                          profound_run <- do.call(profoundMultiBand, all_params)

                          # Create (or replace) catalogue file.
                          if (file.exists(cat_name)) {
                            file.remove(cat_name)
                          }
                          h5createFile(cat_name)

                          # Load the catalogue and add a photometry group.
                          catalogue <- H5Fopen(cat_name)
                          h5createGroup(catalogue, "photometry")
                          group <- H5Gopen(catalogue, "photometry")

                          # For each of the requested outputs.
                          for (data in list(profound_run$cat_tot, profound_run$cat_col, profound_run$cat_grp)) {
                            if (is.null(data) == FALSE) {

                              # Keep only the science image data.
                              sci_columns <- grep("_sci_", names(data), value = TRUE)
                              data_sci <- data[, sci_columns]
                              names(data_sci) <- gsub("_sci_", "_", names(data_sci))

                              # Get list of columns containing fluxes,
                              columns <- list("flux", "sky_mean", "sky_sum", "skyRMS_mean", "skyseg_mean")
                              flux_cols <- list()
                              for (column in columns) {
                                flux_cols <- c(flux_cols, grep(column, names(data_sci), value = TRUE))
                              }
                              flux_cols <- unlist(flux_cols)

                              # and convert to desired flux unit.
                              data_sci[flux_cols] <- data_sci[flux_cols] * config[["flux_conversion"]]

                              # Remove variables that were not calculated.
                              all_na_columns <- colSums(is.na(data_sci)) == nrow(data_sci)
                              data_sci <- data_sci[, !all_na_columns]

                              # Add each column to the hdf5 file.
                              # If no outputs requested, use all.
                              if (is.null((outputs)) == TRUE) {
                                outputs <- names(data_sci)
                              }
                              for (col in names(data_sci)) {
                                if (col %in% outputs) {
                                  h5write(unlist(data_sci[[col]]), file = group, name = col)
                                }
                              }
                            }
                          }

                          # Loop over each attribute in att_config,
                          for (key in names(att_config)) {
                            if ((typeof(att_config[[key]]) == "logical") || (is.null(att_config[[key]]) == TRUE)) {
                              att_config[[key]] <- toString(att_config[[key]])
                            }
                            # Write the attribute to the HDF5 group.
                            h5writeAttribute(attr = att_config[[key]], h5obj = group, name = key)
                          }

                          # Save segmentation images if requested.
                          if (config[["keepsegims"]] == TRUE) {
                            segims <- profound_run$segimlist
                            names <- profound_run$multibands

                            # Function to save each image to its corresponding filename
                            save_image <- function(image_, filename_) {
                              Rfits_write_image(image_, gsub(".hdf5", glue("_{filename_}.fits"), cat_name), keyvalues = hdr)
                            }

                            # Iterate through the lists and save images
                            Map(save_image, segims, names)
                          }

                          H5Fclose(catalogue)
                          H5Gclose(group)
                          h5closeAll()

                        }
                        ))

# Below is designed to be used with Python wrapper profound.py or running
# from the command line. This should be removed if running purely with R.

# Get the command-line arguments.
args <- commandArgs(trailingOnly = TRUE)
arg_list <- parse_args(args)

# Initalise the profound object using the config path.
p_run <- ProFound(arg_list[["config_path"]])

# Get the image paths in the correct format.
if ("img2" %in% names(arg_list)) {
  science <- list(arg_list[["img1"]], arg_list[["img2"]])
} else {
  science <- arg_list[["img1"]]
}

# Get the requested outputs.
if (arg_list[["outputs"]] == "None") {
  outputs <- NULL
} else {
  outputs <- strsplit(outputs, ",")[[1]]
}

# Pass additional parameters.
`%notin%` <- Negate(`%in%`)

parameters <- hash()
for (key in names(arg_list)) {
  if (key %notin% c("img1", "img2", "outputs", "outdir", "config_path")) {
    parameters[[key]] <- arg_list[[key]]
  }
}

# Finally, run ProFound.
p_run$extract(science, parameters, outputs, arg_list[["outdir"]])