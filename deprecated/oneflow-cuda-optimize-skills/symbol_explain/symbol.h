/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_COMMON_SYMBOL_H_
#define ONEFLOW_CORE_COMMON_SYMBOL_H_

#include <mutex>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/check.h"
#include "oneflow/core/common/hash_eq_trait_ptr.h"

namespace oneflow {

template<typename T>
struct SymbolUtil;

// 这段代码定义了一个模板类Symbol,它封装了一个指向类型T的指针 ptr_。
template<typename T>
class Symbol final {
 public:
  // Symbol():构造一个nullptr指针
  Symbol() : ptr_(nullptr) {}
  // Symbol(const T& obj):构造一个指向obj的指针
  Symbol(const T& obj) : ptr_(GetOrCreatePtr(obj)) {}
  // Symbol(const Symbol& rhs):拷贝构造函数,默认实现
  Symbol(const Symbol& rhs) = default;
  // Symbol(Symbol&& rhs):移动构造函数,默认实现
  Symbol(Symbol&& rhs) = default;
  ~Symbol() = default;

  // 显式转换为bool:检查ptr_是否为nullptr
  explicit operator bool() const { return ptr_ != nullptr; }
  // operator->():访问ptr_指向的对象
  const T* operator->() const { return ptr_; }
  // operator*():解引用ptr_指向的对象
  const T& operator*() const { return *ptr_; }
  // operator==():比较两个Symbol的ptr_是否相等
  bool operator==(const Symbol<T>& rhs) const { return ptr_ == rhs.ptr_; }
  // operator!=():对应operator==()
  bool operator!=(const Symbol<T>& rhs) const { return !(*this == rhs); }
  // hash_value():返回ptr_的哈希值
  size_t hash_value() const { return std::hash<const T*>()(ptr_); }

  // Symbol类的拷贝赋值运算符, 将当前Symbol对象的ptr_指针指向other Symbol对象的ptr_指向的对象。
  Symbol& operator=(const Symbol& other) {
    ptr_ = other.ptr_;
    return *this;
  }
  // reset():将ptr_置为nullptr
  void reset() { ptr_ = nullptr; }
  // reset(const T& obj):将ptr_指向obj
  void reset(const T& obj) { ptr_ = GetOrCreatePtr(obj); }

  // shared_from_symbol():返回ptr_指向对象的shared_ptr
  const std::shared_ptr<const T>& shared_from_symbol() const;

 private:
  template<typename SymbolT>
  friend struct SymbolUtil;
  // 私有静态方法GetOrCreatePtr(const T& obj):此方法会返回obj的指针,如果该指针不存在会先创建它。
  static const T* GetOrCreatePtr(const T& obj);

  const T* ptr_;
};

// 定义IsScalarType<Symbol<T>>的目的就是为了指定Symbol<T>应当被视为一个标量类型,就像int, double等内置类型一样,
// 可以用于期望标量类型的上下文。这增加了Symbol类的通用性,使其在更多场景下可以像内置类型一样使用。
// 比如：
// cpp
// void func(auto x) {
//   if constexpr (IsScalarType<decltype(x)>::value) {
//     // ...
//   }
// }

// int main() {
//   func(1);        // IsScalarType<int> is true, so we enter the if branch 
//   func(Symbol<int>(1)); // IsScalarType<Symbol<int>> is also true, so we enter the if branch
// }

template<typename T>
struct IsScalarType<Symbol<T>> final {
  static const bool value = true;
};

// SymbolUtil是一个 final 结构,它为Symbol类提供了静态方法来管理指向T类型对象的共享指针。它包含以下主要功能:
template<typename T>
struct SymbolUtil final {
  using SymbolMap = std::unordered_map<HashEqTraitPtr<const T>, std::shared_ptr<const T>>;

  // GlobalSymbolMap():返回一个全局static SymbolMap,用于存储共享指针
  static SymbolMap* GlobalSymbolMap() {
    static SymbolMap symbol_map;
    return &symbol_map;
  }

  // GlobalSymbolMapMutex():返回一个全局static mutex,用于同步访问GlobalSymbolMap
  static std::mutex* GlobalSymbolMapMutex() {
    static std::mutex mutex;
    return &mutex;
  }

  // ThreadLocalSymbolMap():返回一个线程本地static SymbolMap
  static SymbolMap* ThreadLocalSymbolMap() {
    static thread_local SymbolMap thread_local_symbol_map;
    return &thread_local_symbol_map;
  }

  // ThreadLocalSymbolPtrSet():返回一个线程本地static unordered_set,存储const T*
  static std::unordered_set<const T*>* ThreadLocalSymbolPtrSet() {
    static thread_local std::unordered_set<const T*> thread_local_symbol_ptr_set;
    return &thread_local_symbol_ptr_set;
  }

  // 静态模板方法LocalThreadGetOr():
  // 此方法首先在ThreadLocalSymbolMap中查找对象obj的共享指针。
  // 如果找到,直接返回;如果未找到,调用模板参数GetIter4ObjectAndHashValue查找全局SymbolMap。
  // 找到后,将其添加到ThreadLocalSymbolMap和ThreadLocalSymbolPtrSet中,并返回共享指针。

  // GetIter4ObjectAndHashValue 是SymbolUtil::LocalThreadGetOr()方法的模板参数。
  // 它代表一个函数,该函数接受一个对象obj和其哈希值hash_value作为参数,并返回该对象在SymbolMap中的迭代器。
  // 也就是说,当LocalThreadGetOr()在线程本地SymbolMap中未找到对象obj时,它会调用这个模板参数
  // GetIter4ObjectAndHashValue去全局SymbolMap中查找该对象。如果找到,就返回其迭代器,
  // 并将该对象添加到线程本地的SymbolMap和SymbolPtrSet中。
  template<typename SymbolMap::iterator (*GetIter4ObjectAndHashValue)(const T&, size_t)>
  static const std::shared_ptr<const T>& LocalThreadGetOr(const T& obj) {
    auto* thread_local_symbol_map = ThreadLocalSymbolMap();
    size_t hash_value = std::hash<T>()(obj);
    HashEqTraitPtr<const T> obj_ptr_wraper(&obj, hash_value);
    const auto& local_iter = thread_local_symbol_map->find(obj_ptr_wraper);
    if (local_iter != thread_local_symbol_map->end()) { return local_iter->second; }
    const auto& iter = GetIter4ObjectAndHashValue(obj, hash_value);
    (*thread_local_symbol_map)[iter->first] = iter->second;
    GLOGCHECK(ThreadLocalSymbolPtrSet()->emplace(iter->second.get()).second);
    return iter->second;
  }

  // 静态方法FindGlobalSymbol():在全局SymbolMap中查找对象obj的共享指针,找到后返回其迭代器。
  static typename SymbolMap::iterator FindGlobalSymbol(const T& obj, size_t hash_value) {
    HashEqTraitPtr<const T> new_obj_ptr_wraper(&obj, hash_value);
    auto* symbol_map = GlobalSymbolMap();
    std::unique_lock<std::mutex> lock(*GlobalSymbolMapMutex());
    const auto& iter = symbol_map->find(new_obj_ptr_wraper);
    GLOGCHECK(iter != symbol_map->end());
    return iter;
  }

  // 静态方法SharedFromObject():调用LocalThreadGetOr<FindGlobalSymbol>获取obj的共享指针
  static const std::shared_ptr<const T>& SharedFromObject(const T& obj) {
    return LocalThreadGetOr<FindGlobalSymbol>(obj);
  }

  // 静态模板方法CreateGlobalSymbol():
  // 该方法首先为obj创建一个共享指针ptr
  // 然后在全局SymbolMap中插入ptr,返回插入后的迭代器
  static typename SymbolMap::iterator CreateGlobalSymbol(const T& obj, size_t hash_value) {
    std::shared_ptr<const T> ptr(new T(obj));
    HashEqTraitPtr<const T> new_obj_ptr_wraper(ptr.get(), hash_value);
    std::unique_lock<std::mutex> lock(*GlobalSymbolMapMutex());
    return GlobalSymbolMap()->emplace(new_obj_ptr_wraper, ptr).first;
  }

  // 静态方法GetOrCreatePtr():调用LocalThreadGetOr<CreateGlobalSymbol>获取obj的共享指针。如果不存在会首先创建。
  static const std::shared_ptr<const T>& GetOrCreatePtr(const T& obj) {
    return LocalThreadGetOr<CreateGlobalSymbol>(obj);
  }
};

// 如果ptr_为nullptr,返回一个静态的空shared_ptr
// 否则,调用SymbolUtil::SharedFromObject获取ptr_指向的对象的共享指针,并返回
// 该方法使我们可以随时获取Symbol管理的对象的共享指针,这增加了Symbol的灵活性。
template<typename T>
const std::shared_ptr<const T>& Symbol<T>::shared_from_symbol() const {
  if (this->ptr_ == nullptr) {
    static auto* none = new std::shared_ptr<const T>();
    return *none;
  }
  return SymbolUtil<T>::SharedFromObject(*this->ptr_);
}

// 调用SymbolUtil::GetOrCreatePtr获取对象obj的指针, 返回获取的指针
// 这是Symbol构造函数中调用的方法,用于确保能获取obj的指针。

template<typename T>
const T* Symbol<T>::GetOrCreatePtr(const T& obj) {
  return SymbolUtil<T>::GetOrCreatePtr(obj).get();
}

// 接受一个对象obj作为参数
// 使用obj构造一个Symbol对象并返回
// 这是一个工厂方法,用于简化Symbol对象的构造。我们可以写Symbol<T> s = SymbolOf(obj);构造一个Symbol对象。
template<typename T>
Symbol<T> SymbolOf(const T& obj) {
  return Symbol<T>(obj);
}

}  // namespace oneflow

namespace std {

// 这段代码为Symbol<T>模板类定义了一个hash结构。它重载了operator()方法,使我们可以直接使用std::hash<Symbol<T>>来获取一个Symbol<T>对象的哈希值。
// 具体来说,该hash结构将运算符()定义为简单地委托到Symbol<T>对象的hash_value()方法。
// 这意味着,当我们调用std::hash<Symbol<T>>(symbol)时,它会直接调用symbol.hash_value()并返回结果。
// cpp
// Symbol<int> s(1);
// size_t hash = std::hash<Symbol<int>>()(s); 
// Equivalent to `size_t hash = s.hash_value();`
// 因为我们定义了这个hash结构,所以现在可以使用std::hash直接获取Symbol对象的哈希值,而不需要直接调用hash_value()方法。
// 这增加了Symbol类的灵活性,使其可以用于更多基于哈希的场景。
// 举个例子,现在我们可以直接将Symbol对象用作std::unordered_map的键:
// cpp
// std::unordered_map<Symbol<int>, std::string> map;
// map[Symbol<int>(1)] = "one";
// 这是因为std::unordered_map要求其键类型可哈希,而我们为Symbol定义的这个hash结构满足了这个需求。
template<typename T>
struct hash<oneflow::Symbol<T>> final {
  size_t operator()(const oneflow::Symbol<T>& symbol) const { return symbol.hash_value(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_SYMBOL_H_
