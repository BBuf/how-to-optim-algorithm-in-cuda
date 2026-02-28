## symbol.h

解析 oneflow 的 symbol 实现。

Symbol类是一个模板类,用于包装类型T的对象指针。它通过std::shared_ptr管理对象,并提供接口来设置/获取管理的对象。

主要功能如下:
1. 构造函数:
- Symbol(const T& obj):使用obj构造Symbol,使用SymbolUtil::GetOrCreatePtr获取obj的共享指针
- Symbol(T* ptr):使用ptr构造Symbol,使用std::shared_ptr<const T>包装ptr
- 拷贝构造函数和赋值运算符:将一个Symbol管理的对象拷贝给另一个Symbol
2. reset():改变Symbol管理的对象,等效于设置一个新的共享指针
3. get():获取Symbol管理的对象的原始指针
4. operator->():arrow operator,代表访问Symbol管理的对象
5. hash_value():获取Symbol管理的对象的哈希值
6. shared_from_symbol():获取Symbol管理的对象的共享指针

该类还定义了IsScalarType<Symbol<T>>为true,这使它可以在需要标量类型的上下文中使用。 
其静态方法主要依赖SymbolUtil来管理T类型对象共享指针的获取/创建。SymbolUtil维护着全局和线程本地的SymbolMap来存储共享指针。
该类也定义了一个hash结构,使其可以用于基于哈希的场景,如std::unordered_map。
所以,总体来说,Symbol类是一个管理类型T对象指针的模板类。它使用std::shared_ptr和SymbolUtil实现了对象指针的共享和重用,并提供了便捷的接口来设置和使用其管理的对象。
这使得Symbol可以像内置指针类型一样灵活和高效地使用,增强了代码的抽象能力

