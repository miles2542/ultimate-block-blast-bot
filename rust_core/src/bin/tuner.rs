// tuner.rs
use rand::RngExt;
use rayon::prelude::*;
use std::time::Instant;

use rust_core::board::Bitboard;
use rust_core::evaluator::Weights;
use rust_core::piece::{common_pieces, PieceMasks};
use rust_core::scoring::ScoreEngine;
use rust_core::search::SearchEngine;

use std::env;

struct TunerConfig {
    population_size: usize,
    generations: usize,
    games_per_individual: usize,
    max_turns_per_game: usize,
    time_limit_ms: u64,
    lookahead_depth: usize,
    lookahead_samples: usize,
    aggressive_mode: bool,
}

#[derive(Clone, Debug)]
struct Genome {
    weights: Weights,
    fitness: f32,
    avg_score: f32,
}

impl Genome {
    fn random() -> Self {
        let mut rng = rand::rng();
        Self {
            weights: Weights {
                hole_penalty: rng.random_range(-100.0..=0.0),
                density_penalty: rng.random_range(-20.0..=0.0),
                score_reward: rng.random_range(0.1..=10.0),
                combo_preservation_reward: rng.random_range(0.0..=500.0),
                bumpiness_penalty: rng.random_range(-20.0..=0.0),
                well_depth_penalty: rng.random_range(-50.0..=0.0),
            },
            fitness: 0.0,
            avg_score: 0.0,
        }
    }

    fn mutate(&mut self) {
        let mut rng = rand::rng();
        // 10% mutation rate per gene
        let mutation_chance = 0.1;
        let mutation_scale = 0.2; // Max 20% swing

        if rng.random_bool(mutation_chance) {
            let swing = (self.weights.hole_penalty.abs() * mutation_scale).max(0.1);
            self.weights.hole_penalty += rng.random_range(-swing..=swing);
        }
        if rng.random_bool(mutation_chance) {
            let swing = (self.weights.density_penalty.abs() * mutation_scale).max(0.1);
            self.weights.density_penalty += rng.random_range(-swing..=swing);
        }
        if rng.random_bool(mutation_chance) {
            let swing = (self.weights.score_reward.abs() * mutation_scale).max(0.1);
            self.weights.score_reward += rng.random_range(-swing..=swing);
        }
        if rng.random_bool(mutation_chance) {
            let swing = (self.weights.combo_preservation_reward.abs() * mutation_scale).max(0.1);
            self.weights.combo_preservation_reward += rng.random_range(-swing..=swing);
        }
        if rng.random_bool(mutation_chance) {
            let swing = (self.weights.bumpiness_penalty.abs() * mutation_scale).max(0.1);
            self.weights.bumpiness_penalty += rng.random_range(-swing..=swing);
        }
        if rng.random_bool(mutation_chance) {
            let swing = (self.weights.well_depth_penalty.abs() * mutation_scale).max(0.1);
            self.weights.well_depth_penalty += rng.random_range(-swing..=swing);
        }
    }

    fn crossover(a: &Self, b: &Self) -> Self {
        let mut rng = rand::rng();
        Self {
            weights: Weights {
                hole_penalty: if rng.random_bool(0.5) {
                    a.weights.hole_penalty
                } else {
                    b.weights.hole_penalty
                },
                density_penalty: if rng.random_bool(0.5) {
                    a.weights.density_penalty
                } else {
                    b.weights.density_penalty
                },
                score_reward: if rng.random_bool(0.5) {
                    a.weights.score_reward
                } else {
                    b.weights.score_reward
                },
                combo_preservation_reward: if rng.random_bool(0.5) {
                    a.weights.combo_preservation_reward
                } else {
                    b.weights.combo_preservation_reward
                },
                bumpiness_penalty: if rng.random_bool(0.5) {
                    a.weights.bumpiness_penalty
                } else {
                    b.weights.bumpiness_penalty
                },
                well_depth_penalty: if rng.random_bool(0.5) {
                    a.weights.well_depth_penalty
                } else {
                    b.weights.well_depth_penalty
                },
            },
            fitness: 0.0,
            avg_score: 0.0,
        }
    }
}

fn simulate_game(
    weights: &Weights,
    all_variants: &[PieceMasks],
    score_engine: &ScoreEngine,
    config: &TunerConfig,
) -> (u32, u32) {
    let mut board = Bitboard::empty();
    let mut combo = 0;
    let mut total_score = 0;
    let mut turns_survived = 0;
    let mut rng = rand::rng();

    for _turn in 0..config.max_turns_per_game {
        let mut turn_success = false;
        let max_rerolls = 10;

        for _reroll in 0..max_rerolls {
            // Pick 3 random piece variants
            let p1 = &all_variants[rng.random_range(0..all_variants.len())];
            let p2 = &all_variants[rng.random_range(0..all_variants.len())];
            let p3 = &all_variants[rng.random_range(0..all_variants.len())];
            let pieces = vec![p1, p2, p3];

            // 1. FAST CHECK: Is this turn survivable? (10ms, depth 0)
            let is_survivable = SearchEngine::find_best_turn(
                board,
                &pieces,
                combo,
                score_engine,
                weights,
                10, // Fast check
                0,  // depth 0
                0,  // samples 0
            )
            .is_some();

            if is_survivable {
                // 2. ACTUAL SEARCH: Complex evaluation (Lookahead if configured)
                if let Some(result) = SearchEngine::find_best_turn(
                    board,
                    &pieces,
                    combo,
                    score_engine,
                    weights,
                    config.time_limit_ms,
                    config.lookahead_depth,
                    config.lookahead_samples,
                ) {
                    total_score += result.points_generated;
                    board = result.final_board;
                    combo = result.final_combo;
                    turns_survived += 1;
                    turn_success = true;
                    break;
                }
            }
        }

        if !turn_success {
            break; // game over
        }
    }

    (total_score, turns_survived)
}

fn evaluate_fitness(
    genome: &mut Genome,
    all_variants: &[PieceMasks],
    score_engine: &ScoreEngine,
    config: &TunerConfig,
) {
    let mut total_fitness = 0.0;
    let mut total_raw_score = 0;

    for _ in 0..config.games_per_individual {
        let (score, turns) = simulate_game(&genome.weights, all_variants, score_engine, config);
        total_raw_score += score;

        if config.aggressive_mode {
            // Score Velocity: Average points per turn. Encourages explosive clears.
            let velocity = if turns > 0 {
                score as f32 / turns as f32
            } else {
                0.0
            };
            total_fitness += velocity;
        } else {
            // Safe Endurance: Absolute Max score.
            total_fitness += score as f32;
        }
    }

    genome.fitness = total_fitness / (config.games_per_individual as f32);
    genome.avg_score = total_raw_score as f32 / (config.games_per_individual as f32);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let aggressive = args.iter().any(|arg| arg == "--aggressive");

    let config = if aggressive {
        TunerConfig {
            population_size: 50,
            generations: 30,
            games_per_individual: 5,
            max_turns_per_game: 150,
            time_limit_ms: 300,
            lookahead_depth: 1,
            lookahead_samples: 3,
            aggressive_mode: true,
        }
    } else {
        TunerConfig {
            population_size: 100,
            generations: 30,
            games_per_individual: 10,
            max_turns_per_game: 150,
            time_limit_ms: 100,
            lookahead_depth: 0,
            lookahead_samples: 0,
            aggressive_mode: false,
        }
    };

    println!("=== Block Blast Bot: Genetic Heuristic Tuner ===");
    println!(
        "Mode: {}",
        if config.aggressive_mode {
            "AGGRESSIVE (Expectimax Lookahead)"
        } else {
            "SAFE ENDURANCE (Static)"
        }
    );
    println!("Initializing population of {}", config.population_size);

    let all_variants = common_pieces::get_all_variants();
    let score_engine = ScoreEngine::new();

    let mut population: Vec<Genome> = (0..config.population_size)
        .map(|_| Genome::random())
        .collect();

    for gen in 1..=config.generations {
        let start_time = Instant::now();

        // 1. Evaluate Fitness (Parallel processing using Rayon!)
        population.par_iter_mut().for_each(|genome| {
            evaluate_fitness(genome, &all_variants, &score_engine, &config);
        });

        // 2. Sort by fitness descending
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let best = &population[0];
        let exec_time = start_time.elapsed().as_secs_f32();

        println!(
            "Gen {:03} | Best Fit: {:8.1} | Avg Score: {:8.0} | Time: {:.2}s | Weights: H:{:.1}, D:{:.2}, S:{:.1}, C:{:.1}, B:{:.2}, W:{:.1}",
            gen, best.fitness, best.avg_score, exec_time,
            best.weights.hole_penalty, best.weights.density_penalty,
            best.weights.score_reward, best.weights.combo_preservation_reward,
            best.weights.bumpiness_penalty, best.weights.well_depth_penalty,
        );

        // 3. Selection, Crossover & Mutation
        let mut next_generation = Vec::with_capacity(config.population_size);

        // Elite selection (keep top 20%)
        let elite_count = config.population_size / 5;
        for i in 0..elite_count {
            next_generation.push(population[i].clone());
        }

        // Generate rest through crossover of top 50%
        let mut rng = rand::rng();
        let crossover_pool_size = config.population_size / 2;

        while next_generation.len() < config.population_size {
            let parent_a = &population[rng.random_range(0..crossover_pool_size)];
            let parent_b = &population[rng.random_range(0..crossover_pool_size)];

            let mut child = Genome::crossover(parent_a, parent_b);
            child.mutate();
            next_generation.push(child);
        }

        population = next_generation;
    }

    println!("\n=== Tuning Complete! ===");
    println!("Top 5 Optimal Weights (Consensus Check):");

    for i in 0..5 {
        let genome = &population[i];
        println!(
            "{} | Fit: {:.1} | Score: {:.0} | H:{:.1}, D:{:.2}, S:{:.1}, C:{:.1}, B:{:.2}, W:{:.1}",
            i + 1,
            genome.fitness,
            genome.avg_score,
            genome.weights.hole_penalty,
            genome.weights.density_penalty,
            genome.weights.score_reward,
            genome.weights.combo_preservation_reward,
            genome.weights.bumpiness_penalty,
            genome.weights.well_depth_penalty,
        );
    }
}
