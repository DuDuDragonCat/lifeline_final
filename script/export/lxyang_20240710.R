FOLDERPATH_SQL <- "~/GitHub/lifeline_final/data/SQLite"
setwd(FOLDERPATH_SQL)
# readRDS()
library(tidyverse)
library(RSQLite)
library(jsonlite)
library(pbapply)
library(parallel)
con <- dbConnect(RSQLite::SQLite(), file.path(FOLDERPATH_SQL, "firstcall.db"))
trans_df <- dbGetQuery(con, "SELECT * FROM trans")
esos_df <- dbGetQuery(con, "SELECT * FROM esos")
dbDisconnect(con)
cl <- makeCluster(16L)
clusterEvalQ(cl, library(tidyverse))
clusterEvalQ(cl, library(jsonlite))
trans_list <- pblapply(split(trans_df, seq(nrow(trans_df))), function(raw_row) {
  # raw_row = trans_df[1,]
  trans_text_df <- fromJSON(raw_row$trans_json)
  trans_text_list <- lapply(split(trans_text_df, seq(nrow(trans_text_df))), function(text_row) {
    # text_row = trans_text_df[1,]
    text_clean_df <- tibble(
      tagger = unlist(text_row$tagger_raw),
      pos = unlist(text_row$pos_raw)
    ) %>%
      filter(!stringr::str_detect(pos, "CATEGORY$")) %>%
      # 去除空白
      filter(!stringr::str_detect(pos, "WHITESPACE"))
    retList <- list(
      id = text_row$id,
      seq = text_row$seq,
      tagger_raw = unlist(text_row$tagger_raw),
      pos_raw = unlist(text_row$pos_raw),
      tagger_rm = unlist(text_clean_df$tagger),
      pos_rm = unlist(text_clean_df$pos)
    )
  })
  retList <- list(
    case_no = raw_row$case_no,
    suci_rated = raw_row$suci_rated,
    filename = raw_row$filename,
    trans = trans_text_list
  )
  return(retList)
}, cl = cl)
stopCluster(cl)
# saveRDS(trans_list, file = "trans_list.rds")
# saveRDS(esos_df, file = "esos_df.rds")

# t1 <- readRDS(file = "trans_list.rds")
# e1 <- readRDS(file = "esos_df.rds")
