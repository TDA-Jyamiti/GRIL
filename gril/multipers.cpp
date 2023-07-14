#include "multipers.h"
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <tuple>
#include <future>
#include "utils.h"
#include <omp.h>

#include "./phat/compute_persistence_pairs.h"


Tensor Multipers::compute_l_worm(const int d){
    
    assert (l > 0);
    auto tensopt = torch::TensorOptions().device(torch::kCPU);

    // compute anchor points. There should be 4*(2l-1) anchor points
    int num_anchor_pts = 8*l - 4;
    std::vector<Point> anchor_pts (num_anchor_pts, std::make_pair(0, 0));
    // bottom right: (4l-1)th anchor pt
    auto br = std::make_pair(l * d, - l * d);
    anchor_pts[num_anchor_pts - 1] = br;

    auto br_1 = std::make_pair(br.first - 2 * d, br.second);
    auto br_2 = std::make_pair(br.first, br.second+ 2*d);
    // Iteration over 2l-1 boxes to figure out 8l-6 anchor pts. After this
    // left will be 2l-1 th anchor pt
    for (auto i = 0; i < 4*l - 3; i++){
        anchor_pts[i] = br_1;
        anchor_pts[num_anchor_pts-2-i] = br_2;
        br_1.first = br_1.first - (i % 2) * d;
        br_1.second = br_1.second + ((i + 1) % 2) * d;
        
        br_2.first = br_2.first - ((i + 1) % 2) * d;
        br_2.second = br_2.second + (i % 2) * d;

        // br_1 = std::make_pair(br_1.first - (i % 2) * d, br_1.second + ((i + 1) % 2) * d);
        // br_2 = std::make_pair(br_2.first, br_2.second);
    }
    anchor_pts[4*l-3] = std::make_pair(br_1.first, br_2.second);
    std::vector<Tensor> grid_pts; 
    for(auto i = 0; i < num_anchor_pts; i++){
        grid_pts.push_back(torch::tensor({anchor_pts[i].first, anchor_pts[i].second}, tensopt));
    }

    // Lower staircase
    
    // for (auto b = 0; b < 2*l-2; b++){
    //     auto cur_pt_x = anchor_pts[b].first;
    //     auto cur_pt_y = anchor_pts[b].second;
    //     grid_pts.push_back(torch::tensor({cur_pt_x, cur_pt_y}, tensopt));
    //     for (auto i = 1; i < 2 * d; i++){
    //         cur_pt_x = cur_pt_x - ((i + 1) % 2);
    //         cur_pt_y = cur_pt_y + (i % 2);
    //         grid_pts.push_back(torch::tensor({cur_pt_x, cur_pt_y}, tensopt));
    //     }
    // }

    // // Box region

    // for (auto b = 2*l-2; b < 2*l; b++){
    //     auto cur_pt_x = anchor_pts[b].first;
    //     auto cur_pt_y = anchor_pts[b].second;
    //     grid_pts.push_back(torch::tensor({cur_pt_x, cur_pt_y}, tensopt));
    //     for (auto i = 1; i < 2 * d; i++){
    //         cur_pt_x = cur_pt_x + (b % 2);
    //         cur_pt_y = cur_pt_y + ((b + 1) % 2);
    //         grid_pts.push_back(torch::tensor({cur_pt_x, cur_pt_y}, tensopt));
    //     }
    // }
    
    // // Upper staircase
    
    // for (auto b = 2*l; b < 4*l -2; b++){
    //     auto cur_pt_x = anchor_pts[b].first;
    //     auto cur_pt_y = anchor_pts[b].second;
    //     grid_pts.push_back(torch::tensor({cur_pt_x, cur_pt_y}, tensopt));
    //     for (auto i = 1; i < 2*d; i++){
    //         cur_pt_x = cur_pt_x + ((i + 1) % 2);
    //         cur_pt_y = cur_pt_y - (i % 2);
    //         grid_pts.push_back(torch::tensor({cur_pt_x, cur_pt_y}, tensopt));
    //     }
    // }

    // grid_pts.push_back(torch::tensor({anchor_pts[4*l-2].first, anchor_pts[4*l-2].second}, tensopt));
    auto grid_points_along_boundary_t = torch::stack(grid_pts, 0);
    return grid_points_along_boundary_t;

}


std::vector<std::tuple<bool, Integer>> Multipers::compute_filtration_along_boundary_cap(const Tensor& grid_pts_along_boundary_t,
                                                                                    const Tensor& f,
                                                                                    const Tensor& f_x_sorted,
                                                                                    const Tensor& f_y_sorted,
                                                                                    const Tensor& f_x_sorted_id,
                                                                                    const Tensor& f_y_sorted_id,
                                                                                    int &manual_birth_pts,
                                                                                    int &manual_death_pts){
    //std::cout << " computing filtration along boundary " << std::endl;
    const auto grid_pts_along_boundary = grid_pts_along_boundary_t.to(f.device());
    std::vector<std::tuple<bool, Integer>> simplices_birth_death;
    auto num_grid_pts_along_boundary = grid_pts_along_boundary.size(0);
    Tensor temp;
    Tensor x_new, x_old, y_new, y_old;
    Tensor x_0, y_0, x_last, y_last;
    bool ch = true;
    x_0 = grid_pts_along_boundary[0][0];
    y_0 = grid_pts_along_boundary[0][1];
    temp = (x_0 >= f.index({"...", 0})) & (y_0 >= f.index({"...", 1}));
    Tensor simplices_born_at_0 = torch::where(temp)[0];
    simplices_birth_death.reserve(simplices_born_at_0.size(0));
    for (auto idx = 0; idx < simplices_born_at_0.size(0); idx++)
        simplices_birth_death.emplace_back(ch, simplices_born_at_0[idx].item<int>());
    manual_birth_pts = simplices_birth_death.size();


    for (auto i=0; i < num_grid_pts_along_boundary- 1; i++) {
        x_new = grid_pts_along_boundary[i + 1][0];
        y_new = grid_pts_along_boundary[i + 1][1];
        x_old = grid_pts_along_boundary[i][0];
        y_old = grid_pts_along_boundary[i][1];

        auto up_condition_t = (x_new == x_old) & (y_new > y_old);
        auto up_condition = up_condition_t.item<bool>();

        auto r_condition_t = (x_new > x_old) & (y_new == y_old);
        auto r_condition = r_condition_t.item<bool>();

        auto l_condition_t = (x_new < x_old) & (y_new == y_old);
        auto l_condition = l_condition_t.item<bool>();

        auto down_condition_t = (x_new == x_old) & (y_new < y_old);
        auto down_condition = down_condition_t.item<bool>();


        // # Up arrow
        if (up_condition) {
            // temp = (y_new >= f.index({"...", 1})) & (x_old >= f.index({"...", 0})) & (y_old < f.index({"...", 1}));
            // ch = true;

            temp = (y_new >= f_y_sorted.index({"...", 1})) & (x_old >= f_y_sorted.index({"...", 0})) & (y_old < f_y_sorted.index({"...", 1}));
            auto idx = torch::nonzero(temp);
            temp = f_y_sorted_id.index({idx});
            ch = true;
        }
        // # Right arrow
        else if (r_condition) {
            // temp = (x_new >= f.index({"...", 0})) & (y_old >= f.index({"...", 1})) & (x_old < f.index({"...", 0}));
            // ch = true;

            temp = (x_new >= f_x_sorted.index({"...", 0})) & (y_old >= f_x_sorted.index({"...", 1})) & (x_old < f_x_sorted.index({"...", 0}));
            auto idx = torch::nonzero(temp);
            temp = f_x_sorted_id.index({idx});
            ch = true;
        }
        // # Left arrow
        else if (l_condition) {
            // temp = (x_new < f.index({"...", 0})) & (y_old >= f.index({"...", 1})) & (x_old >= f.index({"...", 0}));
            // ch = false;

            temp = (x_new < f_x_sorted.index({"...", 0})) & (y_old >= f_x_sorted.index({"...", 1})) & (x_old >= f_x_sorted.index({"...", 0}));
            auto idx = torch::nonzero(temp);
            temp = f_x_sorted_id.index({idx});
            ch = false;
        }
        // # Down arrow
        else if (down_condition) {
            // temp = (y_new < f.index({"...", 1})) & (x_old >= f.index({"...", 0})) & (y_old >= f.index({"...", 1}));
            // ch = false;

            temp = (y_new < f_y_sorted.index({"...", 1})) & (x_old >= f_y_sorted.index({"...", 0})) & (y_old >= f_y_sorted.index({"...", 1}));
            
            auto idx = torch::nonzero(temp);
            temp = f_y_sorted_id.index({idx});
            ch = false;
        }
        
        const auto ind = temp;
        if (ch){
        for (auto idx = 0; idx < ind.size(0); idx++){
                auto simp = ind[idx].item<int>();
                simplices_birth_death.emplace_back(ch, simp);
            }
        }
        else{
            for (auto idx = ind.size(0) - 1; idx >= 0; idx--){
                auto simp = ind[idx].item<int>();
                simplices_birth_death.emplace_back(ch, simp);
            }

        }
    }
        /*
        simplices_birth_death is a dictionary which stores the birth and death times of each simplex occuring in
        the boundary cap as a list where even indices store the birth times and the odd indices store the
        death times.
        */

        // # For Tao's code, manual deletion
        manual_death_pts = (int)simplices_birth_death.size();
        x_last = grid_pts_along_boundary.index({-1})[0];
        y_last = grid_pts_along_boundary.index({-1})[1];
        temp = (x_last >= f.index({"...", 0})) & (y_last >= f.index({"...", 1}));
        const Tensor ind = torch::where(temp)[0];
        ch = false;
        for (auto idx = ind.size(0)-1; idx >= 0; idx--)
            simplices_birth_death.emplace_back(ch, ind[idx].item<int>());
    return simplices_birth_death;

}

void Multipers::zigzag_pairs(std::vector<std::tuple<bool, Integer>> &simplices_birth_death,
                        const vector<Simplex> &simplices, 
                        const int manual_birth_pts, 
                        const int manual_death_pts,
                        std::vector<int> &num_full_bars){

    std::vector<Integer> orig_f_add_id;
    std::vector<Integer> orig_f_del_id;
    std::vector<phat::index> bound_c;
    phat::boundary_matrix< phat::bit_tree_pivot_column > bound_chains;
    const auto filt_len = simplices_birth_death.size();
    bound_chains.set_num_cols(filt_len + 1);
    Integer simp_num = filt_len;
    

    // Add the Omega vertex for the coning
    bound_chains.set_col(0, bound_c);
    bound_chains.set_dim(0, 0);

    orig_f_add_id.reserve(simp_num);
    orig_f_del_id.reserve(simp_num);

    std::vector<Integer> del_ids;
    del_ids.reserve(simp_num);

    auto *p_id_map = new SimplexIdMap();
    SimplexIdMap id_map = *p_id_map;


    Integer orig_f_id = 0;
    std::string line;
    Integer s_id = 1;
    Integer death;


    for(auto i=0; i<filt_len; i++) {

        const std::tuple<bool, Integer> record = simplices_birth_death[i];
        char op = std::get<0>(record);
        Integer simplex_id = std::get<1>(record);
        const auto& simp = simplices[simplex_id];
        // std::cout << "i: " << "simplex_id: " << simplex_id << simp << std::endl;
        if (op) {
            getBoundaryChainPhat(id_map, simp, bound_c);            
            bound_chains.set_col(s_id, bound_c);
            bound_chains.set_dim(s_id, getDim(bound_c));
            id_map[simp] = s_id;
            orig_f_add_id.push_back(orig_f_id);
            s_id ++;
        } 
        else {
                        
            del_ids.push_back(id_map[simp]);
            orig_f_del_id.push_back(orig_f_id);
        }
        orig_f_id ++;
    }

    assert(del_ids.size() == s_id-1);
    delete p_id_map;
    

    simp_num = del_ids.size();
    assert(simp_num * 2 == filt_len);

    std::vector<Integer> cone_sid(simp_num+1);
    Integer dim;

    for (auto del_id_it = del_ids.rbegin(); del_id_it != del_ids.rend(); ++del_id_it) {
        bound_c.clear();
        bound_c.push_back(*del_id_it);


        std::vector<phat::index> orig_bound_c;
        bound_chains.get_col(*del_id_it, orig_bound_c);

        if (orig_bound_c.size() == 0) {
            bound_c.push_back(0);
        } else {
            for (auto bsimp : orig_bound_c) {
                bound_c.push_back(cone_sid[bsimp]);
            }
        }

        std::sort(bound_c.begin(), bound_c.end());
        bound_chains.set_col(s_id, bound_c);
        bound_chains.set_dim(s_id, getDim(bound_c));

        cone_sid[*del_id_it] = s_id;

        s_id ++;
    }
    

    phat::persistence_pairs pairs;
    phat::compute_persistence_pairs< phat::twist_reduction >( pairs, bound_chains );
    // int num_full_bars = 0;
    for (phat::index idx = 0; idx < pairs.get_num_pairs(); idx++) {
        Integer b = pairs.get_pair(idx).first;
        Integer d = pairs.get_pair(idx).second - 1;
        Integer p = bound_chains.get_dim(b);
        if (d < simp_num) { 
            mapOrdIntv(b, d, orig_f_add_id); 
        }
        else { 
            mapRelExtIntv(p, b, d, orig_f_add_id, orig_f_del_id, simp_num); 
        }
        
        if (b <= manual_birth_pts && d >= manual_death_pts){
            if(p < 2)
                num_full_bars[p] = num_full_bars[p] + 1;
        }
            
    }

}


void Multipers::num_full_bars_for_specific_d(const Tensor& filtration,
                                            const Tensor& f_x_sorted,
                                            const Tensor& f_y_sorted,
                                            const Tensor& f_x_sorted_id,
                                            const Tensor& f_y_sorted_id,
                                            const vector<Simplex>& simplices, 
                                            const Point& p, 
                                            int d, 
                                            std::vector<int> &num_full_bars){
    int d1, d2;
    auto start = std::chrono::high_resolution_clock::now();
    auto bd_cap = compute_l_worm(d);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    
    auto tensopt_real = torch::TensorOptions().dtype(filtration.dtype()).device(filtration.device());
    
    // TODO: Needs bd_cap to be scaled and translated [DONE].
    const auto shift = torch::tensor({p.first, p.second}, tensopt_real);
    const auto scale = torch::tensor({res, res}, tensopt_real);
    const auto lower_left_corner = torch::tensor({ll_x, ll_y}, tensopt_real);
    bd_cap = ((bd_cap + shift) * scale) + lower_left_corner;

    start = std::chrono::high_resolution_clock::now();
    auto ret = compute_filtration_along_boundary_cap(bd_cap, 
                                                    filtration, 
                                                    f_x_sorted,
                                                    f_y_sorted,
                                                    f_x_sorted_id,
                                                    f_y_sorted_id, 
                                                    d1, 
                                                    d2);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (ret.empty()){
            // return 0;
            return;
    }
    
    
    start = std::chrono::high_resolution_clock::now();
    
    zigzag_pairs(ret, simplices, d1, d2, num_full_bars);
    
    end = std::chrono::high_resolution_clock::now();
    
}

Tensor Multipers::find_maximal_worm_for_rank_k(const Tensor &filtration, 
                                            const Tensor& f_x_sorted,
                                            const Tensor& f_y_sorted,
                                            const Tensor& f_x_sorted_id,
                                            const Tensor& f_y_sorted_id,
                                            const vector<Simplex> &simplices, 
                                            const Point &p, 
                                            const int rank, 
                                            std::vector<std::map<int, int>*> rank_info)
{
    auto tensopt_real = torch::TensorOptions().dtype(filtration.dtype()).device(filtration.device());
    int d_max = (int) (1.0 / res) + 1;
    int d_min = 1;
    // std::cout << "Point: " << (p.first * grid_resolution_x) << " , " << (p.second * grid_resolution_x) << std::endl;
    int d = 1;
    int ans = 0;
    // auto rank_info = (this->hom_rank == 0) ? rank_info_h0 : rank_info_h1;
    int num_full_bars_for_this_k;
    while(d_min <= d_max){
        d = (d_min + d_max) / 2;        
        if(rank_info[this->hom_rank]->count(d) == 0){
            std::vector<int> num_full_bars = {0, 0};
            num_full_bars_for_specific_d(filtration,  
                                        f_x_sorted,
                                        f_y_sorted,
                                        f_x_sorted_id,
                                        f_y_sorted_id, 
                                        simplices,
                                        p, 
                                        d, 
                                        num_full_bars);

            (*rank_info[0])[d] = num_full_bars[0];
            (*rank_info[1])[d] = num_full_bars[1];
            num_full_bars_for_this_k = num_full_bars[this->hom_rank];
        }
        else{
            num_full_bars_for_this_k = rank_info[this->hom_rank]->at(d);
        }
        // std::cout <<" Rank = " << num_full_bars_for_this_k << " d = " << d << " Query rank = "<< rank << std::endl;
        if(num_full_bars_for_this_k >= rank){
            ans = d;
            d_min = d + 1;
        }
        else
            d_max = d - 1;
    }
    // auto scale = (grid_resolution_x * grid_resolution_x) + (grid_resolution_y * grid_resolution_y);
    // auto scaled_res = (double)res * std::sqrt(scale);
    auto scaled_res = (double)ans * res;
    // std::cout <<"Landscape val " << scaled_res << " Query rank = "<< rank << std::endl;
    auto res_t = torch::tensor({scaled_res}, tensopt_real);
    
    return res_t;
}


std::vector<Tensor> Multipers::compute_landscape(const std::vector<Point>& pts, const std::vector<std::tuple<Tensor, vector<Simplex>>> &batch){
    /* INPUT: [[p_1, p_2, p_3, ..., p_m], (f, e, num_vertices)]
     * f: The whole filtration
     * e: Boundary edges
     * num_vertices: Number of vertices.
     * [p1, p2, ..., p_m] is the sampled centre-point for grids.
     */

    /*
     * Suppose the result is ret. ret should have size of N where N is the BATCH_SIZE.
     * ret[0] should contain vector<vector<Tensor>>. So that, ret[0][0] has landscapes for rank (k) = 1
     * ret[0][1] has landscapes for rank = 2. size(ret[0][0]) == size(ret[0][1]) == # of sampled points.
     */
    std::vector<size_t> landscape_end_indices;
    
    landscape_end_indices.reserve(batch.size() + 1);
    landscape_end_indices.push_back(0);
    auto point_in_each_batch = std::vector<size_t>();
    point_in_each_batch.reserve(batch.size());
    const auto num_points = pts.size();
    const auto num_ranks = ranks.size();
    auto futures = std::vector<Tensor>(num_points * num_ranks, torch::tensor(0));
    auto start = std::chrono::high_resolution_clock::now();
    // int max_threads = 8;
    auto arg = batch[0];
    const auto& filtration = std::get<0>(arg);
    const auto& simplices = std::get<1>(arg);

    this->set_grid_resolution_and_lower_left_corner(filtration);
    
    int p = 1;
    
    at::set_num_threads(this->max_threads);
    // int max_jobs = omp_get_max_threads();
    // std::cout << "Max threads = " << max_jobs << std::endl;
    const auto f_x_sorted_id = filtration.index({"...", 0}).argsort();
    const auto f_y_sorted_id = filtration.index({"...", 1}).argsort();
    const auto f_x_sorted = filtration.index({f_x_sorted_id});
    const auto f_y_sorted = filtration.index({f_y_sorted_id});
    // std::cout << "f_x: " << f_x_sorted << std::endl;
    // std::cout << "f: " << filtration << std::endl;
    
    int j = 0;
    
    #pragma omp parallel private(j) firstprivate(filtration, f_x_sorted, f_y_sorted, f_x_sorted_id, f_y_sorted_id) shared(num_points, num_ranks, simplices, pts, ranks, futures, hom_rank, std::cout) //default(none) shared(num_points, num_ranks, num_vertices, filtration, simplices, pts, ranks, futures, hom_rank, std::cout)
    {
        #pragma omp for
        for(auto i = 0; i < num_points; i++) {
            for (j = 0; j < num_ranks; j++){
                const auto &point = pts[i];
                const auto &rank = ranks[j];
                int k = j + num_ranks * i;
                //#pragma omp critical
                //{
                    // std::cout << "Num threads = " << omp_get_num_threads() << std::endl;
                    // int tid = omp_get_thread_num();
                    // std::cout << "i: " << i << " j: " << j << std::endl;
                    // std::cout << "Hello from thread " << tid << " for iteration i: " << i << " rk: " << j << std::endl;
                    // std::cout << "flattened k " << k << std::endl;
                    // std::cout << "hom_rank " << hom_rank << std::endl;
                //}
                std::vector<std::map<int, int>*> rank_info = {this->rank_info_h0[i], this->rank_info_h1[i]};
                auto ftr = find_maximal_worm_for_rank_k(filtration, f_x_sorted, f_y_sorted, f_x_sorted_id, f_y_sorted_id, simplices, point, rank, rank_info);
                futures[k] = ftr;
                
            }
        }
    }
    point_in_each_batch.push_back(num_points);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::partial_sum(point_in_each_batch.begin(), point_in_each_batch.end(), landscape_end_indices.begin(), std::plus<>());
    auto ret = std::vector<Tensor>();
    size_t start_id = 0;
    int count = 0;
    for (unsigned long end_id : landscape_end_indices){
        auto l = std::vector<Tensor>();
        for(auto i=start_id; i < end_id; i++){
            std::vector<Tensor> f;
            const auto num_ranks = ranks.size();
            f.reserve(num_ranks);
            for(auto j= 0; j < num_ranks; j++){
                auto f1 = futures[count];
                f.push_back(f1);
                count++;

            }
            const auto f_t = torch::cat(f, -1);
            l.push_back(f_t.unsqueeze(0));
        }
        const auto l_t = torch::cat(l, 0);
        ret.push_back(l_t);
        start_id = end_id;
    }
   
    std::cout << "Took " << duration.count() << " ms" << std::endl;
    
    return ret;
}
// Tensor Multipers::compute_grad_matrix(const std::vector<Point>& pts, const std::vector<std::tuple<Tensor, vector<Simplex>>> &batch){
//     const auto num_points = pts.size();
//     at::set_num_threads(this->max_threads);
    
//     auto arg = batch[0];
//     const auto& filtration = std::get<0>(arg);
//     const auto& simplices = std::get<1>(arg);
//     const auto f_x = filtration.index({"...", 0});
//     const auto f_y = filtration.index({"...", 1});
//     const int l = this->l;
//     const auto tensopt_real = torch::TensorOptions().dtype(filtration.dtype()).device(torch::kCPU);
//     const auto shift = torch::arange(-l, l+1, tensopt_real);
//     // (torch.isclose(5 copies of f_x, x-lines) && f_x <= px) && (f_y <= py + l * d) line. Assign -1 to them. 
//     // torch.isclose(5 copies of f_x, x-lines) && f_x > px. Assign +1 to them.

//     for(auto i = 0; i < num_points; i++){
//         const auto &point = pts[i];
        

//     }


// }

void Multipers::set_grid_resolution_and_lower_left_corner(const Tensor& filtration){
    
    const auto f_min = std::get<0>(filtration.min(0));
    // this->ll_x = f_min.index({0}).item<float>();
    // this->ll_y = f_min.index({1}).item<float>();
    this->ll_x = 0.0;
    this->ll_y = 0.0;
    // std::cout << "dx " << delta_x << " dy " << delta_y << " ll_x " << ll_x << " ll_y " << ll_y << std::endl;
    // std::cout << "f_max " << f_max << " f_min " << f_min << std::endl;
    // std::cout << " ll_x " << ll_x << " ll_y " << ll_y << " f_min " << f_min << std::endl;

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Multipers>(m, "Multipers")
        .def(py::init<int, int, double, int, std::vector<int>>())
        .def("compute_landscape", &Multipers::compute_landscape)
        .def("set_hom_rank", &Multipers::set_hom_rank)
        .def("refresh_rank_info", &Multipers::refresh_rank_info)
        .def("set_max_jobs", &Multipers::set_max_jobs);
}
