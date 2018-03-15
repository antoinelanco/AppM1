//
// Created by samuel on 02/02/18.
//

#include "res.h"
#include <vector>
#include "string_utils.h"

std::string getResFolder() {
    std::string this_file = __FILE__;
    std::vector<std::string> splittedPath = split(this_file, '/');

    while (splittedPath.back() != "AppM1") {
        splittedPath.pop_back();
    }

    std::string res_path;
    for (auto& elt : splittedPath) {
        res_path += "/" + elt;
    }

    res_path += "/res";

    return res_path;
}
