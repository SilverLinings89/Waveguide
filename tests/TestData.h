#pragma once

class TestData {
public:
  unsigned int Inner_Element_Order;
  unsigned int Cells_Per_Direction;

  TestData(unsigned int in_inner_element, unsigned int in_cells) {
    Inner_Element_Order = in_inner_element;
    Cells_Per_Direction = in_cells;
  }

  TestData(const TestData &in_td) {
    Inner_Element_Order = in_td.Inner_Element_Order;
    Cells_Per_Direction = in_td.Cells_Per_Direction;
  }
};