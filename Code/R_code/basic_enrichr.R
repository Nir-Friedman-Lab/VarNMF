utils::setRepositories(ind = 0, addURLs = c(CRAN = "https://cloud.r-project.org/"))

args = commandArgs(trailingOnly=TRUE)

if(!require("optparse")){install.packages("optparse")}
library("optparse")

option_list = list(
  make_option(c("-g", "--genes.txt"), type="character", default=NULL, 
              help="txt file with gene names", metavar="character"),
  make_option(c("-o", "--output.path"), type="character", default="./", 
              help="output directory  [default %default]", metavar="character"),
  make_option(c("-t", "--threshold"), type="double", default=0.05, 
              help="threshold of adjusted pvalue  [default %default]", metavar="double"),
  make_option(c("-f", "--outputfilename"), type="character", default="enrichr.output.csv", 
              help="output file name [default %default]", metavar="character")
); 

args_parser = OptionParser(option_list=option_list);
args = parse_args(args_parser);

if (is.null(args$genes.txt)){
  print_help(args_parser)
  stop("At least one argument must be supplied (genes file path).txt", call.=FALSE)
}


if(!require("enrichR")){install.packages("enrichR")}
library(enrichR,quietly = T)

enricher <- function(genes.txt,
                     output.path="./",
                     outputfilename="enrichr.output.csv",
                     threshold=0.05){
  genes <- unlist(read.csv(genes.txt,header=F))
  setEnrichrSite("Enrichr")
  dbs.we.want <-c("Reactome_2022" ,"BioPlanet_2019","Cancer_Cell_Line_Encyclopedia","Human_Gene_Atlas",
                 "ARCHS4_Tissues", "ARCHS4_Cell-lines","PanglaoDB_Augmented_2021", "HuBMAP_ASCTplusB_augmented_2022")
  en <- do.call(rbind.data.frame,enrichr(genes,dbs.we.want))
  en <- en[en$Adjusted.P.value < threshold, ]
  write.csv(en, paste0(output.path,outputfilename))
}

enricher(args$genes.txt,
         args$output.path,
         args$outputfilename,
         args$threshold)



