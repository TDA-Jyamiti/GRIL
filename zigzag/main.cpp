#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <tuple>
#include <future>
#include "utils.h"
#include "./phat/compute_persistence_pairs.h"
#include "./torchph/chofer_torchex/pershom/pershom_cpp_src/vertex_filtration_comp_cuda.h"
#include "./torchph/chofer_torchex/pershom/pershom_cpp_src/vr_comp_cuda.cuh"
#include "./torchph/chofer_torchex/pershom/pershom_cpp_src/calc_pers_cuda.cuh"



using torch::Tensor;

inline Integer getDim(const std::vector<phat::index> &bound_c) {
    if (bound_c.empty()) { return 0; }
    return (Integer)bound_c.size() - 1;
}

void getBoundaryChainPhat(const SimplexIdMap &id_map,
                          const Simplex &simp, std::vector<phat::index> &bound_c) {
    bound_c.clear();
    if (simp.size() <= 1) { return; }
    bound_c.reserve(simp.size());

    Simplex bound_simp(simp.begin()+1, simp.end());
    bound_c.push_back(id_map.at(bound_simp));
    // std::cout << "  " << bound_simp << endl;

    for (Integer i = 0; i < simp.size()-1; ++i) {
        bound_simp[i] = simp[i];
        // std::cout << "  " << bound_simp << endl;
        bound_c.push_back(id_map.at(bound_simp));
    }

    std::sort(bound_c.begin(), bound_c.end());
}

void parseSimplex(const Tensor& simplex, Simplex& simp){
    auto dim = simplex.size(0);
    for(auto i = 0; i < dim; i++){
        simp.push_back(simplex[i].item<int>());
    }
}
inline void mapOrdIntv(Integer &b, Integer &d, std::vector<Integer>& orig_f_add_id) {
    // assert(b-1 > 0);
    // assert(d < orig_f_add_id.size());

    // Up-down interval is same,
    // so directly map to interval of input filtration
    b = orig_f_add_id[b-1] + 1;
    d = orig_f_add_id[d];
}


inline void mapRelExtIntv(Integer &p, Integer &b, Integer &d, std::vector<Integer>& orig_f_add_id, std::vector<Integer>& orig_f_del_id, int simp_num) {
    // assert(d >= simp_num);

    if (b > simp_num) { // Open-closed
        // Map to up-down interval
        std::swap(b, d);
        b = 3 * simp_num - b;
        d = 3 * simp_num - d;
        p --;

        // Map to interval of input filtration
        b = orig_f_del_id[b-1-simp_num] + 1;
        d = orig_f_del_id[d-simp_num];
    } else { // Closed-closed
        // Map to up-down interval
        d = 3 * simp_num - d-1;

        // Map to interval of input filtration
        b = orig_f_add_id[b-1];
        d = orig_f_del_id[d-simp_num];

        if (b < d) {
            b = b + 1;
        } else {
            std::swap(b, d);
            b = b + 1;
            p = p - 1;
        }
    }
}

Tensor calculate_zigzag(const std::vector<std::tuple<Tensor, char>> &records, const Tensor &filtration, const int max_dim)
{
    auto ret = std::vector<Tensor>();
    auto tensopt_real = torch::TensorOptions()
            .dtype(filtration.dtype())
            .device(filtration.device());

    auto tensopt_int = torch::TensorOptions()
            .dtype(torch::kInt64)
            .device(filtration.device());

    std::vector<int64_t> orig_f_add_id;
    std::vector<int64_t> orig_f_del_id;

    auto simp_num = filtration.size(0);
    //std::cout << "Computing zz " << std::endl;

    std::vector<phat::index> bound_c;
    auto *bound_chains = new phat::boundary_matrix< phat::bit_tree_pivot_column >();
    auto filt_len = (long long)records.size();
    bound_chains->set_num_cols(filt_len + 1);

    // create boundary array ...
    auto ba = torch::empty(
            {filt_len + 1,
             (max_dim+1) * 2},
            tensopt_int
    );
    ba.fill_(-1);

    // Add the Omega vertex for the coning
    bound_chains->set_col(0, bound_c);
    bound_chains->set_dim(0, 0);


    orig_f_add_id.reserve(simp_num);
    orig_f_del_id.reserve(simp_num);

    std::vector<Integer> del_ids;
    del_ids.reserve(simp_num);

    auto *p_id_map = new SimplexIdMap ();
    SimplexIdMap id_map = *p_id_map;


    int64_t orig_f_id = 0;
    std::string line;
    Integer s_id = 1;
    Integer death;

    std::vector<int64_t> num_simplices_by_dim;
    // Add cone vertex dimension
    num_simplices_by_dim.push_back(0);
    Simplex simp;
    for(auto i = 0; i < filt_len; i++) {
        simp.clear();
        const auto& record = records[i];
        char op = std::get<1>(record);
        const auto &simplex = std::get<0>(record);
        parseSimplex(simplex, simp);
        if (op == 'i') {
            getBoundaryChainPhat(id_map, simp, bound_c);
            const auto dim = getDim(bound_c);
            bound_chains->set_col(s_id, bound_c);
            bound_chains->set_dim(s_id, dim);
//            std::cout << "Simplex: " << simp << " s_id: " << s_id << std::endl;
//            std::cout << "Boundary: ";
            std::reverse(bound_c.begin(), bound_c.end());
            for(auto j = 0; j < bound_c.size(); j++) {
                ba.index_put_({s_id, j}, bound_c[j]);
//                std::cout << bound_c[j] << " ";
            }
//            std::cout << std::endl;
//            std::cout << ba << std::endl;

            num_simplices_by_dim.push_back(dim);

            id_map[simp] = s_id;
            orig_f_add_id.push_back(orig_f_id);
            s_id ++;
        } else {
            // TODO: del the entry in 'id_map'?


            del_ids.push_back(id_map[simp]);
            orig_f_del_id.push_back(orig_f_id);
        }

        orig_f_id ++;
    }

    assert(del_ids.size() == s_id-1);
    delete p_id_map;

    // assert(bound_chains.size() == simp_num+1);
    // assert(orig_f_add_id.size() == simp_num);
    // assert(orig_f_del_id.size() == simp_num);
    // assert(del_ids.size() == simp_num);

    simp_num = del_ids.size();
    assert(simp_num * 2 == filt_len);

    // std::vector<Integer> cone_sid(simp_num+1, -1);
    std::vector<Integer> cone_sid(simp_num+1);
    // Chain bound_c;


    for (auto del_id_it = del_ids.rbegin(); del_id_it != del_ids.rend(); ++del_id_it) {
        bound_c.clear();
        bound_c.push_back(*del_id_it);

        // std::cout << std::endl << bound_chains[*del_id_it] << std::endl;

        std::vector<phat::index> orig_bound_c;
        bound_chains->get_col(*del_id_it, orig_bound_c);

        // if (bound_chains[*del_id_it].size() == 0) {
        if (orig_bound_c.empty()) {
            bound_c.push_back(0);
        } else {
            // for (auto bsimp : bound_chains[*del_id_it]) {
            for (auto bsimp : orig_bound_c) {
                // assert(cone_sid[bsimp] >= 0);
                bound_c.push_back(cone_sid[bsimp]);
            }
        }

        std::sort(bound_c.begin(), bound_c.end());
        // std::cout << s_id << ": " << *del_id_it << " " << bound_c << std::endl;
        const auto dim = getDim(bound_c);
        // ComputingPersistenceForSimplicialMapElementary(bound_c);
        bound_chains->set_col(s_id, bound_c);
        bound_chains->set_dim(s_id, dim);

//        std::cout << "Simplex: " << simp << " s_id: " << s_id << std::endl;
//        std::cout << "Boundary: ";

        std::reverse(bound_c.begin(), bound_c.end());
        for(auto j = 0; j < bound_c.size(); j++) {
            ba.index_put_({s_id, j}, bound_c[j]);
//            std::cout << bound_c[j] << " ";
        }
//        std::cout << std::endl;
        num_simplices_by_dim.push_back(dim);
        cone_sid[*del_id_it] = s_id;
        s_id ++;

    }
    delete bound_chains;
//    auto num_simplices_by_dim_t = torch::tensor::z
    Tensor simplex_dim = torch::tensor(num_simplices_by_dim, tensopt_int);

    // compressing
    auto i = simplex_dim.gt(0).nonzero().squeeze();
    auto ba_row_i_to_bm_col_i = torch::arange(0, ba.size(0), tensopt_int);

    ba_row_i_to_bm_col_i = ba_row_i_to_bm_col_i.index_select(0, i);
    ba = ba.index_select(0, i);

    // ret.push_back(ba);
    // ret.push_back(ba_row_i_to_bm_col_i);
    // ret.push_back(simplex_dim);
    // std::cout << "BA: " << ba << std::endl;
    // std::cout << "Max dim: " << max_dim << std::endl;
    auto calc_pers_output = CalcPersCuda::calculate_persistence(
            ba, 
            ba_row_i_to_bm_col_i, 
            simplex_dim, 
            max_dim, 
            -1 
        );
        // VRCompCuda::calculate_persistence_output_to_barcode_tensors
        // read_barcode_from_birth_death_times
        // return VRCompCuda::calculate_persistence_output_to_barcode_tensors(
        //     calc_pers_output, 
        //     filtration
        // );
        // std::cout << "Pers out: " << calc_pers_output << std::endl;
        std::vector<Tensor> bd_pairs;
        for (auto ii = 0; ii <= max_dim; ii++){
            auto h_dim = calc_pers_output[0][ii].to(torch::kLong);
            //print(f'Before dim appending {h_dim}')
            if (h_dim.size(0) != 0){
                auto dim_vec = torch::ones({h_dim.size(0), 1}, tensopt_int) * ii;
                auto pbd_bar = torch::cat({dim_vec, h_dim}, 1);
                bd_pairs.push_back(pbd_bar);
            }
        }
        auto bd_pairs_t = torch::cat(bd_pairs, 0);
        auto open_int = torch::zeros_like(bd_pairs_t);
        open_int.index_put_({"...", 2}, 1);
        bd_pairs_t = bd_pairs_t - open_int;
        auto orig_f_add_id_t = torch::tensor(orig_f_add_id, tensopt_int);
        auto orig_f_del_id_t = torch::tensor(orig_f_del_id, tensopt_int);
        bd_pairs_t = map_bars(bd_pairs_t, orig_f_add_id_t, orig_f_del_id_t, simp_num);
        return bd_pairs_t;
    
    // ret.push_back(orig_f_add_id_t);
    // ret.push_back(orig_f_del_id_t);

    // auto ret_tuple = std::make_tuple(ret, simp_num);
    // return ret_tuple;

}
/*
std::vector<std::vector<Tensor>> zigzag_persistence_single(const std::vector<std::tuple<Tensor, char>> &records, const Tensor &filtration, const int max_dim){
    const auto ret = calculate_persistence_args(records, filtration, max_dim);
    const auto &ba = ret.at(0);
    const auto &ba_row_i_to_bm_col_i = ret.at(1);
    const auto &simplex_dim = ret.at(2);

    auto calc_pers_output = CalcPersCuda::calculate_persistence(
            ba,
            ba_row_i_to_bm_col_i,
            simplex_dim,
            max_dim,
            -1
    );
    auto barcodes = VRCompCuda::calculate_persistence_output_to_barcode_tensors(
            calc_pers_output,
            filtration
    );
    for(const auto &pers: barcodes){
        for(auto dim = 0; dim < max_dim; dim++){
            std::cout << "h" << dim << "-essential:\n" << pers[0][dim] << std::endl;
            std::cout << "h" << dim << "-non-essential:\n" << pers[1][dim] << std::endl;
        }

    }
    return barcodes;
}
*/

Tensor test(){
    auto tensopt_int = torch::TensorOptions()
            .dtype(torch::kInt64)
            .device(torch::kCUDA);
    std::vector<std::tuple<Tensor, char>> records_t;
    std::vector<std::tuple<Simplex, char>> records;
    std::vector<int> filtration;
    std::vector<Simplex> smp;
    smp = {{0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}};
    records.emplace_back(smp[0], 'i');
    records.emplace_back(smp[1], 'i');
    records.emplace_back(smp[2], 'i');
    records.emplace_back(smp[3], 'i');
    records.emplace_back(smp[4], 'i');
    records.emplace_back(smp[5], 'i');
    records.emplace_back(smp[5], 'd');
    records.emplace_back(smp[5], 'i');
    records.emplace_back(smp[3], 'd');
    records.emplace_back(smp[5], 'd');
    records.emplace_back(smp[4], 'd');
    records.emplace_back(smp[1], 'd');
    records.emplace_back(smp[2], 'd');
    records.emplace_back(smp[0], 'd');
    for (const auto &record: records){
        const auto simp = std::get<0>(record);
        const auto op = std::get<1>(record);
        Tensor s = torch::tensor(simp);
        records_t.emplace_back(s, op);
    }
    auto num_filt = (int64_t)records.size();
    const auto filt = torch::arange(1, num_filt + 1, tensopt_int);
    auto ret = calculate_zigzag(records_t, filt, 2);
    // auto ret = zigzag_persistence_single(records_t, filt, 2);
//    auto test_ret = std::make_tuple(ret, smp.size() + 1);
    
    return ret;

}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("calculate_zigzag", &calculate_zigzag, "Computes boundary matrices for fast-zigzag");
    m.def("test", &test, "Performs single test");
}

