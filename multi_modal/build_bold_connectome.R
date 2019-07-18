library(optparse)
library(colormap)
library(ANTsR)

parser <- OptionParser()
parser <- add_option(parser, c("-v", "--verbose"),
                    action="store_true",
                    default=T, help='print extra output [default]')
parser <- add_option(parser, c("-c", "--cleanbold"), type="character")
args <- parse_args(parser)

print(args$cleanbold)
