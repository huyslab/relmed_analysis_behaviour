### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 70f082e6-ff53-11ef-2c76-01abbcd2749e
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, Printf, Combinatorics, JuMP, HiGHS, AlgebraOfGraphics, Images, ImageTransformations
	using LogExpFunctions: logistic, logit
end

# ╔═╡ 5f827d58-fbb2-42d3-ac79-7c9b1502198f
# Parameters
begin
	n_categories_PILT = 21
	n_categories_WM = 36
	n_categories_LTM = 12

	n_sessions = 6

	n_extra = 50

end

# ╔═╡ fe25d625-bbbe-4466-83c5-559c62827554
begin
	# Load memoraribility data
	mem_image = DataFrame(CSV.File("./generate_experimental_sequences/THINGS_Memorability_Scores.csv"))
	
	mem_category = DataFrame(CSV.File("./generate_experimental_sequences/Concept_to_category_linking.csv"))
end

# ╔═╡ cd55b869-7383-4c1b-9375-c28aef91c9d8
unique(mem_category.category_label)

# ╔═╡ 1cc6ff99-aa17-40d5-84e8-4ce61c3e049c
begin

	# From ChatGPT
	danger_concepts = ["aircraft_carrier", "airplane", "alligator", "ambulance", "armor", "arrow", "axe", "barbed_wire", "bazooka", "bear", "bomb", "brass_knuckles", "bullet", "bulletproof_vest", "cannon", "cannonball", "catapult", "chainsaw", "coffin", "crossbow", "dagger", "detonator", "dynamite", "electric_chair", "fire", "firecracker", "flamethrower", "gallows", "grenade", "guillotine", "gun", "hawk", "hearse", "helicopter", "helmet", "humvee", "javelin", "jet", "jeep", "knife", "landmine", "machete", "minefield", "missile", "mortar", "mummy", "musketeer", "pistol", "rifle", "rocket", "saber", "scythe", "shield", "shuriken", "skull", "spear", "sword", "tank", "target", "tombstone", "torpedo", "trap", "trident"]

	money_concepts = ["bank", "cash_machine", "cash_register", "checkbook", "coin", "credit_card"]

	disgusting_concepts = [
	    "ant",
	    "barnacle",
	    "bat1",
	    "bat2",
	    "bee",
	    "beetle",
	    "bug",
	    "cockroach",
	    "compost",
	    "crab",
	    "denture",
	    "diaper",
	    "dogfood",
	    "dough",
	    "drain",
	    "earwig",
	    "eel",
	    "fly",
	    "flypaper",
	    "flyswatter",
	    "fungus",
	    "garbage",
	    "gravy",
	    "hotdog",
	    "jellyfish",
	    "leech",
	    "lobster",
	    "maggot",
	    "manhole",
		"mosquito",
	    "moth",
	    "mucus",
	    "mussel",
	    "octopus",
	    "oyster",
	    "phlegm",
	    "rat",
	    "rotting_food",
	    "scab",
	    "sea_cucumber",
	    "slug",
	    "snail",
	    "snot",
	    "squid",
	    "tarantula",
		"tick",
	    "toenail",
	    "trash",
	    "urinal",
	    "vomit",
	    "worm"
	]

	scary_concepts = ["altar", "anvil", "bat1", "bat2", "beehive", "bee", "bug", "bulletin_board", "cage", "cobra", "cockroach", "cross", "crow", "crowbar", "crucifix", "dagger", "dart", "eel", "fire_alarm", "fire_hydrant", "fire_pit", "fireplace", "firetruck", "firewood", "fireworks", "fungus", "gargoyle", "grave", "gravestone", "guillotine", "hammer", "hook1", "hook2", "hyena", "jail", "knife", "lock", "maggot", "mask", "moth", "mousetrap", "noose", "pit", "pitchfork", "poison", "potion", "rat", "rattlesnake", "raven", "rope", "scorpion", "skeleton", "skull", "spider", "stake", "storm", "tarantula", "thorn", "tombstone", "trap", "vulture", "wasp", "witch", "wolf", "zombie"]

	manual_removed = ["woman", "girl", "boy", "marijuana", "pill", "pocketknife", "sewage", "slot_machine", "banner", "cigar", "cigarette", "cigarette_butt", "crystal", "footrest", "gas_mask", "holster", "man", "mannequin", "metal_detector", "money", "mullet", "pillbox", "pipe1", "police_car", "shell1", "sickle", "sonogram", "splinter", "squirt_gun", "stilt", "toilet", "water_heater", "wine", "wooden_leg", "whoopee_cushion", "wallet", "toilet_paper", "syringe", "string_cheese", "slime", "sheath", "shower_cap", "riser", "retainer", "pothole", "piggy_bank", "petri_dish", "patty", "mouthpiece", "mold2", "lip_balm", "lighter", "hole", "handcuff", "hairnet", "fishhook", "dumpster", "crystal1", "cornhusk", "contact_lens", "coffee_filter", "chicken1", "bedpan", "baby", "air_mattress", "pickax", "barrette", "blindfold", "bologna", "braid", "catfish", "curling_iron", "harness", "hula_hoop", "mascara", "ready_meal", "sausage", "spam", "tattoo", "wig", "badge", "dreidel"]

	manual_removed_images = ["hot_tub_14s.jpg", "snorkel_05s.jpg", "hot_tub_10s.jpg", "walrus_12s.jpg", "tee_11s.jpg", "teabag_12s.jpg", "peg_05s.jpg", "subway_08s.jpg"]

end

# ╔═╡ 8e36d09b-d153-48ea-8f04-2c63a5c5866e
# Number of categories needed
n_total_categories = (n_categories_PILT + n_categories_WM + n_categories_LTM) * 
	n_sessions + n_extra

# ╔═╡ 59d0eb9a-cdbb-4f62-a631-f4808f5ba38e
function select_top_n(df::DataFrame, N::Int)
    # Compute category counts
    category_counts = combine(groupby(df, :category_label), nrow => :count)

    # Compute the proportion of each category
    total_rows = sum(category_counts.count)
    category_counts.proportion = category_counts.count ./ total_rows

    # Allocate samples proportionally
    category_counts.samples = round.(Int, category_counts.proportion .* N)

    # Ensure total samples sum to N (adjust due to rounding errors)
    while sum(category_counts.samples) ≠ N
        diff = N - sum(category_counts.samples)
        idx = argmax(category_counts.proportion)  # Adjust the largest proportion
        category_counts.samples[idx] += sign(diff)

    end

    # Select top rows within each category
    selected = DataFrame()
    for row in eachrow(category_counts)
        cat = row.category_label
        k = row.samples
        subset = filter(r -> r.category_label == cat, df)
        sorted_subset = sort(subset, :CR, rev=true)
        append!(selected, sorted_subset[1:min(k, nrow(sorted_subset)), :])
    end

    return selected
end

# ╔═╡ 4a85eb23-5e7d-461e-ae92-ab078979ac43
# Select categories
selected_concepts = let
	# Remove bad categories
	filter!(x -> (x.concept ∉ vcat(
		danger_concepts, 
		money_concepts, 
		disgusting_concepts,
		scary_concepts,
		manual_removed
	)) && 
		(x.category_label ∉ ["weapon", "body part", "clothing", "clothing accessory", "medical equipment", "part of car"]) && (!isnan(x.CR)), mem_category)

	# Select relevant columns and computer pctile for CR
	select!(
		mem_category,
		:concept,
		:category_label,
		:CR,
		:CR => (x -> (ecdf(x)).(x) * 100) => :CR_pcntile
	)

	# Choose concepts with highest score, sampling from meta categories
	selected_concepts = select_top_n(mem_category, n_total_categories)
end

# ╔═╡ 11484465-a457-4103-80f5-38823c5afaed
# Pick top images from each category
selected_images = let rng = Xoshiro(0)

	# Filter manually removed images
	filter!(x -> x.image_name ∉ manual_removed_images, mem_image)

	# Exctract concept from image name
	mem_image.concept = (s -> s[1:last(findlast('_', s))-1]).(mem_image.image_name)

	# Join concepts with images
	concept_image = innerjoin(
		mem_image,
		selected_concepts,
		on = :concept
	)

	@assert length(unique(concept_image.concept)) == nrow(selected_concepts)

	# Get CR ranking within each concept
	DataFrames.transform!(
		groupby(concept_image, :concept),
		:cr => (x -> sortperm(x, rev = true)) => :cr_rank
	)

	# Select top two images from each category
	selected_images = filter(
		x -> x.cr_rank < 3,
		concept_image
	)

	@assert length(unique(selected_images.concept)) == nrow(selected_concepts)
	@assert nrow(selected_images) == n_total_categories * 2

	# Rename images, assign 1 and 2 randomly
	DataFrames.transform!(
		groupby(selected_images, :concept),
		:concept => (x -> shuffle(rng, 1:2)) => :filename
	)

	DataFrames.transform!(
		selected_images,
		[:concept, :filename] => ByRow((c, r) -> "$(c)_$r.jpg") => :filename
	)

	selected_images.file_path = (x -> replace(x, "images/" => "generate_experimental_sequences/THINGS/")).(selected_images.file_path)

	selected_images
end

# ╔═╡ e600d12a-57cc-4084-bd7e-07e090320534
# Copy images to one folder, save csv
let
	dest_folder = "generate_experimental_sequences/trial1_stimuli/"

	for file in readdir(dest_folder; join=true)
        rm(file; recursive=true)
    end
	
	for (file, name) in zip(selected_images.file_path, selected_images.filename)
	    cp(file, joinpath(dest_folder, name), force=true)
	end

	CSV.write("generate_experimental_sequences/trial1_stimuli/stimuli.csv", selected_images)

end

# ╔═╡ aee2f526-f89a-41d7-a27b-3b1cad2b0203
# Resize images
let
	# Set the folder paths
	folder = "generate_experimental_sequences/trial1_stimuli/"
			
	# Resize and save each image
	for file in selected_images.filename
	    img = load(joinpath(folder, file))     # Load image
	    resized_img = imresize(img, (500, 500))  # Resize to 500x500
	    save(joinpath(folder, file), resized_img)  # Save resized image
	end

end

# ╔═╡ Cell order:
# ╠═70f082e6-ff53-11ef-2c76-01abbcd2749e
# ╠═5f827d58-fbb2-42d3-ac79-7c9b1502198f
# ╠═fe25d625-bbbe-4466-83c5-559c62827554
# ╠═cd55b869-7383-4c1b-9375-c28aef91c9d8
# ╠═1cc6ff99-aa17-40d5-84e8-4ce61c3e049c
# ╠═8e36d09b-d153-48ea-8f04-2c63a5c5866e
# ╠═4a85eb23-5e7d-461e-ae92-ab078979ac43
# ╠═11484465-a457-4103-80f5-38823c5afaed
# ╠═e600d12a-57cc-4084-bd7e-07e090320534
# ╠═aee2f526-f89a-41d7-a27b-3b1cad2b0203
# ╠═59d0eb9a-cdbb-4f62-a631-f4808f5ba38e
