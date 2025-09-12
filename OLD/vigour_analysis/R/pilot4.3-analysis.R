


# Load libraries ----------------------------------------------------------

library(tidyverse)
library(afex)
library(emmeans)
library(here)
set_sum_contrasts()
theme_set(theme_minimal())


# Read data ---------------------------------------------------------------

vigour_4 <- read_csv(here("data", "pilot4_raw_vigour_data.csv"))
vigour_4.3 <- read_csv(here("data", "pilot4.3_raw_vigour_data.csv"))

vigour_4_triallist <- distinct(vigour_4, trial_number, ratio, magnitude, trial_duration)
vigour_4.3_triallist <- distinct(vigour_4.3, trial_number, ratio, magnitude, trial_duration)

all.equal(vigour_4_triallist, vigour_4.3_triallist)


# Remove too-many-missing-trial participants ------------------------------

vigour_4 <- vigour_4 %>%
    group_by(prolific_id) %>%
    mutate(n_missing = sum(trial_presses == 0)) %>%
    ungroup() %>%
    filter(n_missing < 9)

vigour_4.3 <- vigour_4.3 %>%
    group_by(prolific_id) %>%
    mutate(n_missing = sum(trial_presses == 0)) %>%
    ungroup() %>%
    filter(n_missing < 9)

# Reward sensitivity curve ------------------------------------------------

vigour_4 %>%
    group_by(prolific_id, reward_per_press) %>%
    summarize(press_per_sec = mean(press_per_sec)) %>%
    ungroup() %>%
    ggplot(aes(x = reward_per_press, y = press_per_sec)) +
    stat_summary(fun.data = mean_se, geom = "line", aes(color = "4.0")) +
    stat_summary(fun.data = mean_se, geom = "pointrange", aes(color = "4.0")) +
    stat_summary(
        fun.data = mean_se,
        geom = "line",
        aes(color = "4.3"),
        data = vigour_4.3 %>%
            group_by(prolific_id, reward_per_press) %>%
            summarize(press_per_sec = mean(press_per_sec)) %>%
            ungroup()
    ) +
    stat_summary(
        fun.data = mean_se,
        geom = "pointrange",
        aes(color = "4.3"),
        data = vigour_4.3 %>%
            group_by(prolific_id, reward_per_press) %>%
            summarize(press_per_sec = mean(press_per_sec)) %>%
            ungroup()
    )

# Reliability -------------------------------------------------------------

vigour_4.3 %>%
    mutate(block = (trial_number - 1) %/% 9 + 1) %>%
    mutate(rpp_grp = if_else(reward_per_press < 0.5, "low_rpp", "high_rpp")) %>%
    pivot_wider(
        id_cols = c(prolific_id, version, block),
        names_from = rpp_grp,
        values_from = press_per_sec,
        values_fn = mean
    ) %>%
    mutate(rpp_diff = high_rpp - low_rpp) %>%
    pivot_wider(
        id_cols = c(prolific_id, version),
        names_from = block,
        names_prefix = "block",
        values_from = rpp_diff
    ) %>%
    mutate(
        first_half = (block1 + block2) / 2,
        second_half = (block3 + block4) / 2,
        even_half = (block4 + block2) / 2,
        odd_half = (block1 + block3) / 2
    ) %>%
    summarize(
        first_second_cor = cor(first_half, second_half),
        even_odd_cor = cor(even_half, odd_half)
    ) %>%
    mutate(across(everything(), ~ 2 * .x / (1 + .x)))


# PIT data ----------------------------------------------------------------

pit_4 <- read_csv(here("data", "pilot4_raw_pit_data.csv"))

pit_4 <- pit_4 %>%
    group_by(prolific_id) %>%
    mutate(n_missing = sum(trial_presses == 0)) %>%
    ungroup() %>%
    filter(n_missing < 9)
