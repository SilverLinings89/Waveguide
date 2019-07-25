//
// Created by kraft on 25.07.19.
//

#include "../Code/Core/DOFManager.h"

#include "gtest/gtest.h"

TEST(DOFManagerTests, StaticFunctionTest) {
    ASSERT_EQ(DOFManager::testValue(),4);
}
