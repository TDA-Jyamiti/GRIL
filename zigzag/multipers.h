#ifndef MULTIPERS_H
#define MULTIPERS_H
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <tuple>
#include <future>
#include "utils.h"

#include "./phat/compute_persistence_pairs.h"

using torch::Tensor;
using namespace torch::indexing;
typedef std::pair<int, int> Point;

class Multipers{
    private:
        int hom_rank;
        std::vector<int> ranks;
        double step, ll_x, ll_y;
        int px, py;
        int l;
        
        
        void set_hom_rank(int hom_rank){
            this->hom_rank = hom_rank;
        }
        void set_ranks(std::vector<int> ranks_){
            this->ranks.insert(this->ranks.begin(), ranks_.begin(), ranks_.end());
        }
        void set_step(double step){
            this->step = step;
        }
        void set_l_for_worm(int l){
            this->l = l;
        }
        Tensor compute_l_worm(const int d);
        std::vector<std::tuple<bool, Integer>> compute_filtration_along_boundary_cap(const Tensor& grid_pts_along_boundary_t,
                                                                                    const Tensor& f,
                                                                                    int &manual_birth_pts,
                                                                                    int &manual_death_pts);

        int zigzag_pairs(std::vector<std::tuple<bool, Integer>> &simplices_birth_death,
                        const vector<Simplex> &simplices, 
                        const int manual_birth_pts, 
                        const int manual_death_pts);
        
        int num_full_bars_for_specific_d(const Tensor& filtration, const vector<Simplex>& simplices, const Point& p, int d);

        Tensor find_maximal_worm_for_rank_k(const Tensor& filtration, const vector<Simplex>& simplices, const Point& p, const int rank);
        
        void set_grid_resolution_and_lower_left_corner(const Tensor& filtration);


    
    public:
        int max_threads;
        Multipers(const int hom_rank, const int l, double step, const std::vector<int> ranks){
            set_hom_rank(hom_rank);
            set_l_for_worm(l);
            // set_division_along_axes(px, py);
            set_step(step);
            set_ranks(ranks);
            this->max_threads = 1;
        }
        void set_max_jobs(int max_jobs){
            this->max_threads = max_jobs;
        }
        std::vector<Tensor> compute_landscape(const std::vector<Point>& pts, const std::vector<std::tuple<Tensor, vector<Simplex>>> &batch);



};

#endif