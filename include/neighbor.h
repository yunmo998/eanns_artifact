// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include "utils.h"
#include <immintrin.h> 

namespace diskann
{

struct Neighbor
{
    unsigned id;
    float distance;
    bool expanded;

    Neighbor() = default;

    Neighbor(unsigned id, float distance) : id{id}, distance{distance}, expanded(false)
    {
    }

    inline bool operator<(const Neighbor &other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const Neighbor &other) const
    {
        return (id == other.id);
    }
};

// Invariant: after every `insert` and `closest_unexpanded()`, `_cur` points to
//            the first Neighbor which is unexpanded.
class NeighborPriorityQueue
{
  public:
    NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0)
    {
    }

    explicit NeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1)
    {
    }

    // explicit NeighborPriorityQueue(size_t backup_size, size_t capacity) : _size(0), _backup_size(backup_size),
    // _capacity(capacity), _cur(0), _data(backup_size + 1)
    // {
    // }

    // Inserts the item ordered into the set up to the sets capacity.
    // The item will be dropped if it is the same id as an exiting
    // set item or it has a greated distance than the final
    // item in the set. The set cursor that is used to pop() the
    // next item will be set to the lowest index of an uncheck item
    void insert(const Neighbor &nbr)
    {
        if (_size == _capacity && _data[_size - 1] < nbr)
        {
            return;
        }

        size_t lo = 0, hi = _size;
        while (lo < hi)
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid])
            {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            }
            else if (_data[mid].id == nbr.id)
            {
                return;
            }
            else
            {
                lo = mid + 1;
            }
        }

        if (lo < _capacity)
        {
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
        }
        _data[lo] = {nbr.id, nbr.distance};
        if (_size < _capacity)
        {
            _size++;
        }
        if (lo < _cur)
        {
            _cur = lo;
        }
    }

    bool insert_with_ret(const Neighbor &nbr)
    {
        if (_size == _capacity && _data[_size - 1] < nbr)
        {
            return false;
        }

        size_t lo = 0, hi = _size;
        while (lo < hi)
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid])
            {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            }
            else if (_data[mid].id == nbr.id)
            {
                return false;
            }
            else
            {
                lo = mid + 1;
            }
        }

        if (lo < _capacity)
        {
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
        }
        _data[lo] = {nbr.id, nbr.distance};
        if (_size < _capacity)
        {
            _size++;
        }
        if (lo < _cur)
        {
            _cur = lo;
        }
        return true;
    }

    void set_unexpanded(size_t pos)
    {
        if (pos < _size)
        {
            _data[pos].expanded = false;
            if (pos < _cur)
                _cur = pos;
        }
    }

    bool find_and_set_unexpanded(const Neighbor &nbr)
    {
        size_t lo = 0, hi = _size;
        while (lo < hi)
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid])
            {
                hi = mid;
            }
            else if (_data[mid].distance == nbr.distance) // already inserted
            {
                _data[mid].expanded = false;
                // assert(_data[mid].distance == nbr.distance);
                if (mid < _cur)
                    _cur = mid;
                return true;
            }
            else
            {
                lo = mid + 1;
            }
        }
        return false;
    }

    
    bool find_and_set_unexpanded(const uint32_t &id)
    {
        for(size_t i = 0; i < _size; i++)
        {
            if (_data[i].id == id)
            {
                _data[i].expanded = false;
                if (i < _cur)
                    _cur = i;
                return true;
            }
        }
        return false;
    }


    bool find_and_remove(uint32_t id)
    {
        for (size_t i = 0; i < _size; i++)
        {
            if (_data[i].id == id)
            {
                std::memmove(&_data[i], &_data[i + 1], (_size - i) * sizeof(Neighbor));
                _size--;
                return true;
            }
        }
        return false;
    }

    Neighbor closest_unexpanded()
    {
        _data[_cur].expanded = true;
        size_t pre = _cur;
        while (_cur < _size && _data[_cur].expanded)
        {
            _cur++;
        }
        return _data[pre];
    }

    Neighbor closest_unexpanded_prefetch(int stride = 0)
    {
        size_t pos = _cur;
        while (pos < _size && stride > 0)
        {
            if (!_data[pos].expanded)
            {
                if (--stride == 0)
                {
                    return _data[pos];
                }
            }
            ++pos;
        }
        return _data[_cur];
    }

    float last_distance() const
    {
        return _size > 0 ? _data[_size - 1].distance : std::numeric_limits<float>::max();
    }
    
    bool has_unexpanded_node() const
    {
        return _cur < _size;
    }
    // bool has_unexpanded_node_backup() const
    // {
    //     return _cur < _size && _cur < _capacity;
    // }

    bool is_full() const
    {
        return _size == _capacity;
    }

    size_t size() const
    {
        return _size;
    }

    size_t frontier_offset() const
    {
        return _cur;
    }

    size_t capacity() const
    {
        return _capacity;
    }

    void reserve(size_t capacity)
    {
        if (capacity + 1 > _data.size())
        {
            _data.resize(capacity + 1);
        }
        _capacity = capacity;
    }

    Neighbor &operator[](size_t i)
    {
        return _data[i];
    }

    Neighbor operator[](size_t i) const
    {
        return _data[i];
    }

    void clear()
    {
        _size = 0;
        _cur = 0;
    }

    void reset(size_t new_size)
    {
        _size = new_size;
        _cur = 0;
    }

  private:
    size_t _size, _capacity, _cur;
    // size_t _backup_size; // used when _backup_size is different from _capacity, the neighbors with index >= _capacity
    // are not returned by search
    std::vector<Neighbor> _data;
};

} // namespace diskann
