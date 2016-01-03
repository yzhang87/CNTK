# CNTK Coding Style

This page documents the conventions used in the source code of CNTK. Please adhere to these conventions 
when writing new code. Follow common sens and break up functions exceeding a reasonable limit 
(a couple of screen pages), use meaningful names, comment well and keep comments and code in sync, etc.

## Basics: indentation, spacing and braces

Code is consistently indented using four spaces. Tab characters are not allowed anywhere in the code. 
The only exception is makefiles, other build system, or data files where tab characters are syntactically 
required.

The following things are indented:

 * Bodies of control statements: for, if, while, switch, etc.
 * Free statement blocks, i.e. opening and closing braces which do not follow any control statement. These 
 are sometimes used to limit the lifetime of objects.
 * Bodies of classes and functions. 
 * Statements continued from the previous line.
 * Code in case statements starts on the line following the case statement and is indented.

The following things are not indented:
* Contents of namespaces.
* Case labels
* Access control specifiers.

Code is written using Allman or BSD Unix style braces. This style puts the brace associated with a 
control statement on the next line, indented to the same level as the control statement. 
Statements within the braces are indented to the next level. 

Braces are mandatory and must not be omitted even when the body of the block has only a single line.

Spaces are present in the following places:
* Around all binary operators, including assignments
* Between a keyword and parentheses
* Between an identifier or keyword and a brace
* After commas and semicolons that do not end a line
* Between the template keyword and the template argument list

Spaces are absent in the following places:
* Before semicolons and commas
* On the inner side of parentheses
* Between a function name and its argument list
* Between unary operators and their operands
* Inside an empty argument list
* Between a label and a colon
* Around the scope operator ::

Member initializer lists and base class lists that contain more than one class should be written using 
Boost-style indentation: each member or class goes on a separate line and is preceded by the appropriate 
punctuation and a space. This makes it very easy to spot errors.

```
namespace Microsoft {
namespace MSR {
namespace CNTK {

Matrix ImplodeSelf(int x);
int ConfuseUs(float y);

class MainPart:
    public Head,
    protected Heart
{
public:
    MainPart():
        m_weight(99),
        m_height(180)
    {}
private:
    void EatMore();
    int m_consume, m_repeater;
};

template <typename Box>
void Inspect(Box &container)
{
    switch (container)
    {
    case 1:
        PrepareIt();
        break;

    case 2:
        Finish();
        break;

    default:
        break;
    }

    for (int i = 0; i < 30; ++i)
    {
        container << EatMore();
    }
    return container;
}

} // namespace CNTK
} // namespace MSR
} // namespace Microsoft
```

## Naming conventions

* Class and namespace names use UpperCamelCase or PascalCase. 
* Names (SQL, CNTK, ...) can stay in all upper cases. 
* Global and public static functions, stack variables, and class members (class variables) use lowerCamelCase. 
* Class member functions (methods) use UpperCamelCase. 
* Macros and constants use UPPER_SNAKE_CASE. 
* Template parameters which are types use UpperCamelCase. 
* Type prefixes, Hungarian notation, etc. are disallowed. Use meaningful suffixes if you need to disambiguate, 
e.g. matrixFloat and matrixDoubleNormalized.

Name prefies
* ```m_``` for member variables
* ```s_``` for static varibales in any context
* ```g_``` for global variables, which should be avoided in the first place (as much as possible)

Variable names should be nouns. Function names should be verbs, with the exception of getters, which can be 
nouns. For instance, a class property called position would have the setter SetPosition() and the getter Position(). 


## Filename conventions

C++ files should have the .cpp extension, while header files should have the .h extension. Spaces and underscores are not allowed. Using numbers in filenames is discouraged.
```
#define GOOD_MACRO(x) x
void CallOut();
unsigned const THE_ANSWER = 42;

class SolveAllProblems 
{
public:
    void DoWhatWeNeed();
    static void SetBugsOff();
    int countReasons;
protected:
    void NeverWorking();
    static void GetReason();
    int internalCounter;
private:
    void InternalNeeds();
    static void ShowReason();
    int countShows;
};

template <typename TypeParam, int numberOfReasons>
void callGlobal(boost::array<TypeParam, numberOfReasons> const &array);
```

## Preprocessor

Conditional compilation using the preprocessor is strongly discouraged, since it leads to code rot. 
Use it only when it is unavoidable, for instance when an optional dependency is used. 
A special case is using conditional compilation to exclude an entire file based on the platform, which is allowed.

In preference to conditionally compiled, platform-specific code, you should aim to write portable code 
which works the same regardless of platform. Using the BOOST libraries can help a lot in this respect. 
If you must use different code depending on the platform, try to encapsulate it in helper functions, 
so that the amount of code that differs between platforms is kept to a minimum.

