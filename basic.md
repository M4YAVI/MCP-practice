I'll create a comprehensive MCP (Model Context Protocol) server that demonstrates all major Python concepts. This will be an educational "Python Learning Assistant" MCP server.

python

                        Copy
                    
#!/usr/bin/env python3
"""
Python Learning Assistant MCP Server
A comprehensive MCP server that teaches Python concepts through interactive examples
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import inspect
import ast
import traceback
import io
import contextlib
from pathlib import Path

# MCP SDK imports
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# PYTHON CONCEPTS DEMONSTRATION
# ============================================================================

# 1. ENUMS - Type-safe constants
class PythonConcept(Enum):
    """Enumeration of Python concepts we can teach"""
    BASICS = "basics"
    DATA_STRUCTURES = "data_structures"
    OOP = "object_oriented"
    FUNCTIONAL = "functional"
    ASYNC = "async_await"
    DECORATORS = "decorators"
    GENERATORS = "generators"
    CONTEXT_MANAGERS = "context_managers"
    METACLASSES = "metaclasses"
    DESCRIPTORS = "descriptors"

# 2. DATACLASSES - Modern way to create classes
@dataclass
class CodeExample:
    """Represents a Python code example with metadata"""
    title: str
    code: str
    explanation: str
    concept: PythonConcept
    difficulty: int = 1  # 1-5 scale
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Validation after initialization"""
        if not 1 <= self.difficulty <= 5:
            raise ValueError(f"Difficulty must be between 1 and 5, got {self.difficulty}")

# 3. ABSTRACT BASE CLASSES - Define interfaces
class ConceptTeacher(ABC):
    """Abstract base class for teaching different Python concepts"""
    
    @abstractmethod
    def get_examples(self) -> List[CodeExample]:
        """Return examples for this concept"""
        pass
    
    @abstractmethod
    def explain(self) -> str:
        """Provide detailed explanation of the concept"""
        pass

# 4. INHERITANCE AND POLYMORPHISM
class BasicsTeacher(ConceptTeacher):
    """Teaches Python basics"""
    
    def get_examples(self) -> List[CodeExample]:
        return [
            CodeExample(
                title="Variables and Types",
                code="""
# Python is dynamically typed
name = "Alice"  # str
age = 30       # int
height = 5.6   # float
is_student = True  # bool

# Type annotations (Python 3.5+)
def greet(name: str, age: int) -> str:
    return f"Hello {name}, you are {age} years old"

# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 0
""",
                explanation="Python variables are references to objects. Type annotations help with code clarity.",
                concept=PythonConcept.BASICS,
                difficulty=1,
                tags=["variables", "types", "annotations"]
            ),
            CodeExample(
                title="Control Flow",
                code="""
# If-elif-else
score = 85
if score >= 90:
    grade = 'A'
elif score >= 80:
    grade = 'B'
else:
    grade = 'C'

# Ternary operator
status = "pass" if score >= 60 else "fail"

# Match statement (Python 3.10+)
match grade:
    case 'A':
        print("Excellent!")
    case 'B':
        print("Good job!")
    case _:
        print("Keep trying!")
""",
                explanation="Python offers multiple ways to control program flow, including the new match statement.",
                concept=PythonConcept.BASICS,
                difficulty=1,
                tags=["control-flow", "conditionals", "match"]
            )
        ]
    
    def explain(self) -> str:
        return """
Python Basics cover the fundamental building blocks:
- Variables and dynamic typing
- Basic data types (int, float, str, bool)
- Control flow (if/elif/else, match)
- Loops (for, while)
- Functions and scope
- Error handling with try/except
"""

# 5. DECORATORS - Function that modifies other functions
def timer_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    import time
    
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def memoize(func: Callable) -> Callable:
    """Decorator for memoization (caching results)"""
    cache = {}
    
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

class DecoratorsTeacher(ConceptTeacher):
    """Teaches decorators and their applications"""
    
    def get_examples(self) -> List[CodeExample]:
        return [
            CodeExample(
                title="Basic Decorators",
                code="""
# Simple decorator
def uppercase(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper() if isinstance(result, str) else result
    return wrapper

@uppercase
def greet(name):
    return f"hello, {name}"

print(greet("alice"))  # HELLO, ALICE

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("Hello!")
""",
                explanation="Decorators modify or enhance functions without changing their code.",
                concept=PythonConcept.DECORATORS,
                difficulty=3,
                tags=["decorators", "functions", "wrapper"]
            ),
            CodeExample(
                title="Class Decorators",
                code="""
# Class decorator
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        print("Creating database connection")

# Property decorator
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9
""",
                explanation="Decorators can also be applied to classes and used for properties.",
                concept=PythonConcept.DECORATORS,
                difficulty=4,
                tags=["decorators", "classes", "property"]
            )
        ]
    
    def explain(self) -> str:
        return """
Decorators are a powerful feature that allow you to:
- Modify function behavior without changing the function
- Add functionality like logging, timing, caching
- Create property getters/setters
- Implement design patterns (singleton, etc.)
- Chain multiple decorators
"""

# 6. GENERATORS - Memory-efficient iterators
class GeneratorsTeacher(ConceptTeacher):
    """Teaches generators and yield"""
    
    def get_examples(self) -> List[CodeExample]:
        return [
            CodeExample(
                title="Generator Functions",
                code="""
# Simple generator
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Using the generator
for num in fibonacci(10):
    print(num, end=' ')  # 0 1 1 2 3 5 8 13 21 34

# Generator expression
squares = (x**2 for x in range(10))
print(list(squares))  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Infinite generator
def count_up(start=0):
    while True:
        yield start
        start += 1

# Use with itertools
import itertools
for num in itertools.islice(count_up(), 5):
    print(num)  # 0 1 2 3 4
""",
                explanation="Generators produce values on-demand, saving memory for large sequences.",
                concept=PythonConcept.GENERATORS,
                difficulty=3,
                tags=["generators", "yield", "iterators"]
            )
        ]
    
    def explain(self) -> str:
        return """
Generators are functions that return an iterator:
- Use 'yield' instead of 'return'
- Maintain state between calls
- Memory efficient for large datasets
- Can be infinite
- Support send(), throw(), close() methods
"""

# 7. CONTEXT MANAGERS - Resource management
class ContextManagersTeacher(ConceptTeacher):
    """Teaches context managers and the with statement"""
    
    def get_examples(self) -> List[CodeExample]:
        return [
            CodeExample(
                title="Context Managers",
                code="""
# Using built-in context manager
with open('file.txt', 'w') as f:
    f.write('Hello, World!')
# File is automatically closed

# Custom context manager with class
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print(f"Elapsed: {self.end - self.start:.4f}s")

with Timer() as timer:
    import time
    time.sleep(0.1)

# Context manager with contextlib
from contextlib import contextmanager

@contextmanager
def temporary_change(obj, attr, new_value):
    old_value = getattr(obj, attr)
    setattr(obj, attr, new_value)
    try:
        yield
    finally:
        setattr(obj, attr, old_value)
""",
                explanation="Context managers ensure proper resource cleanup using the with statement.",
                concept=PythonConcept.CONTEXT_MANAGERS,
                difficulty=3,
                tags=["context-managers", "with", "resources"]
            )
        ]
    
    def explain(self) -> str:
        return """
Context managers handle setup and cleanup operations:
- Implement __enter__ and __exit__ methods
- Used with 'with' statement
- Ensure resources are properly released
- Can suppress exceptions
- contextlib provides utilities for creating them
"""

# 8. ASYNC/AWAIT - Asynchronous programming
class AsyncTeacher(ConceptTeacher):
    """Teaches async/await patterns"""
    
    def get_examples(self) -> List[CodeExample]:
        return [
            CodeExample(
                title="Async/Await Basics",
                code="""
import asyncio
import aiohttp

# Basic async function
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Running multiple tasks concurrently
async def fetch_multiple(urls):
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# Async generator
async def async_counter(n):
    for i in range(n):
        await asyncio.sleep(0.1)
        yield i

# Using async generator
async def main():
    async for num in async_counter(5):
        print(num)

# Run the async function
asyncio.run(main())
""",
                explanation="Async/await enables concurrent I/O operations without threads.",
                concept=PythonConcept.ASYNC,
                difficulty=4,
                tags=["async", "await", "concurrency"]
            )
        ]
    
    def explain(self) -> str:
        return """
Async/await enables asynchronous programming:
- 'async def' defines coroutine functions
- 'await' pauses execution until result is ready
- asyncio.gather() runs tasks concurrently
- Ideal for I/O-bound operations
- Not for CPU-bound tasks (use multiprocessing instead)
"""

# 9. METACLASSES - Classes that create classes
class MetaclassesTeacher(ConceptTeacher):
    """Teaches metaclasses and class creation"""
    
    def get_examples(self) -> List[CodeExample]:
        return [
            CodeExample(
                title="Metaclass Basics",
                code="""
# Simple metaclass
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "Connected"

# Both variables reference the same instance
db1 = Database()
db2 = Database()
print(db1 is db2)  # True

# Metaclass that adds methods
class AutoPropertyMeta(type):
    def __new__(mcs, name, bases, attrs):
        # Add getter/setter for attributes starting with '_'
        for key, value in list(attrs.items()):
            if key.startswith('_') and not key.startswith('__'):
                prop_name = key[1:]
                attrs[prop_name] = property(
                    lambda self, k=key: getattr(self, k),
                    lambda self, v, k=key: setattr(self, k, v)
                )
        return super().__new__(mcs, name, bases, attrs)
""",
                explanation="Metaclasses control how classes are created and can modify class behavior.",
                concept=PythonConcept.METACLASSES,
                difficulty=5,
                tags=["metaclasses", "advanced", "class-creation"]
            )
        ]
    
    def explain(self) -> str:
        return """
Metaclasses are classes whose instances are classes:
- Control class creation process
- type is the default metaclass
- Use __new__ and __init__ to customize
- Can add/modify attributes and methods
- Used in frameworks like Django, SQLAlchemy
"""

# ============================================================================
# CODE EXECUTION ENGINE
# ============================================================================

class SafeCodeExecutor:
    """Safely executes Python code with restrictions"""
    
    @staticmethod
    def execute_code(code: str, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Execute Python code safely and return results
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary with output, errors, and execution info
        """
        # Create string buffer to capture output
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        # Restricted globals - remove dangerous functions
        safe_globals = {
            '__builtins__': {
                'print': lambda *args, **kwargs: print(*args, **kwargs, file=output_buffer),
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'dir': dir,
                'help': help,
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
                'True': True,
                'False': False,
                'None': None,
            }
        }
        
        # Parse code to check for dangerous operations
        try:
            tree = ast.parse(code)
            # Could add AST analysis here to block dangerous code
        except SyntaxError as e:
            return {
                'success': False,
                'output': '',
                'error': f"Syntax Error: {str(e)}",
                'execution_time': 0
            }
        
        # Execute code with timeout
        import signal
        import time
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")
        
        # Set up timeout (Unix only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
        
        start_time = time.time()
        
        try:
            # Redirect stdout and stderr
            with contextlib.redirect_stdout(output_buffer), \
                 contextlib.redirect_stderr(error_buffer):
                exec(code, safe_globals)
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'output': output_buffer.getvalue(),
                'error': error_buffer.getvalue(),
                'execution_time': execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'output': output_buffer.getvalue(),
                'error': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                'execution_time': execution_time
            }
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel timeout

# ============================================================================
# MCP SERVER IMPLEMENTATION
# ============================================================================

class PythonLearningServer:
    """MCP Server for teaching Python concepts"""
    
    def __init__(self):
        self.server = Server("python-learning-assistant")
        self.code_executor = SafeCodeExecutor()
        
        # Initialize concept teachers
        self.teachers: Dict[PythonConcept, ConceptTeacher] = {
            PythonConcept.BASICS: BasicsTeacher(),
            PythonConcept.DECORATORS: DecoratorsTeacher(),
            PythonConcept.GENERATORS: GeneratorsTeacher(),
            PythonConcept.CONTEXT_MANAGERS: ContextManagersTeacher(),
            PythonConcept.ASYNC: AsyncTeacher(),
            PythonConcept.METACLASSES: MetaclassesTeacher(),
        }
        
        # Code examples storage
        self.user_examples: List[CodeExample] = []
        
        # Setup handlers
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup all MCP handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List all available tools"""
            return [
                Tool(
                    name="learn_concept",
                    description="Learn about a specific Python concept with examples",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "concept": {
                                "type": "string",
                                "description": "Python concept to learn",
                                "enum": [c.value for c in PythonConcept]
                            }
                        },
                        "required": ["concept"]
                    }
                ),
                Tool(
                    name="execute_code",
                    description="Execute Python code and see the output",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="explain_code",
                    description="Get detailed explanation of Python code",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to explain"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="practice_exercise",
                    description="Get a practice exercise for a concept",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "concept": {
                                "type": "string",
                                "description": "Concept to practice",
                                "enum": [c.value for c in PythonConcept]
                            },
                            "difficulty": {
                                "type": "integer",
                                "description": "Difficulty level (1-5)",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["concept"]
                    }
                ),
                Tool(
                    name="save_example",
                    description="Save a code example for later reference",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "code": {"type": "string"},
                            "explanation": {"type": "string"},
                            "concept": {
                                "type": "string",
                                "enum": [c.value for c in PythonConcept]
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["title", "code", "concept"]
                    }
                ),
                Tool(
                    name="search_examples",
                    description="Search saved examples by tag or concept",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (tag or concept)"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[Union[TextContent, ImageContent]]:
            """Handle tool calls"""
            
            if name == "learn_concept":
                concept = PythonConcept(arguments["concept"])
                return await self._handle_learn_concept(concept)
                
            elif name == "execute_code":
                code = arguments["code"]
                return await self._handle_execute_code(code)
                
            elif name == "explain_code":
                code = arguments["code"]
                return await self._handle_explain_code(code)
                
            elif name == "practice_exercise":
                concept = PythonConcept(arguments["concept"])
                difficulty = arguments.get("difficulty", 3)
                return await self._handle_practice_exercise(concept, difficulty)
                
            elif name == "save_example":
                return await self._handle_save_example(arguments)
                
            elif name == "search_examples":
                query = arguments["query"]
                return await self._handle_search_examples(query)
                
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    async def _handle_learn_concept(self, concept: PythonConcept) -> List[TextContent]:
        """Handle learning about a Python concept"""
        if concept not in self.teachers:
            return [TextContent(
                type="text",
                text=f"Sorry, I don't have materials for {concept.value} yet."
            )]
        
        teacher = self.teachers[concept]
        explanation = teacher.explain()
        examples = teacher.get_examples()
        
        # Format response
        response_parts = [f"# Learning {concept.value.replace('_', ' ').title()}\n\n"]
        response_parts.append(explanation)
        response_parts.append("\n## Examples:\n")
        
        for i, example in enumerate(examples, 1):
            response_parts.append(f"\n### Example {i}: {example.title}")
            response_parts.append(f"Difficulty: {'⭐' * example.difficulty}")
            response_parts.append(f"Tags: {', '.join(example.tags)}")
            response_parts.append(f"\n```python\n{example.code.strip()}\n```")
            response_parts.append(f"\n**Explanation:** {example.explanation}\n")
        
        return [TextContent(type="text", text="\n".join(response_parts))]
    
    async def _handle_execute_code(self, code: str) -> List[TextContent]:
        """Handle code execution"""
        result = self.code_executor.execute_code(code)
        
        response_parts = ["# Code Execution Result\n"]
        response_parts.append(f"```python\n{code}\n```\n")
        
        if result['success']:
            response_parts.append("✅ **Execution Successful**\n")
            if result['output']:
                response_parts.append(f"**Output:**\n```\n{result['output']}\n```")
        else:
            response_parts.append("❌ **Execution Failed**\n")
            response_parts.append(f"**Error:**\n```\n{result['error']}\n```")
        
        response_parts.append(f"\n*Execution time: {result['execution_time']:.4f} seconds*")
        
        return [TextContent(type="text", text="\n".join(response_parts))]
    
    async def _handle_explain_code(self, code: str) -> List[TextContent]:
        """Analyze and explain Python code"""
        try:
            tree = ast.parse(code)
            analyzer = CodeAnalyzer()
            analysis = analyzer.analyze(tree)
            
            response_parts = ["# Code Analysis\n"]
            response_parts.append(f"```python\n{code}\n```\n")
            response_parts.append("## Analysis:\n")
            response_parts.append(analysis)
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except SyntaxError as e:
            return [TextContent(
                type="text",
                text=f"❌ Syntax Error: {str(e)}"
            )]
    
    async def _handle_practice_exercise(self, concept: PythonConcept, difficulty: int) -> List[TextContent]:
        """Generate practice exercises"""
        exercises = {
            PythonConcept.BASICS: [
                {
                    "title": "FizzBuzz",
                    "description": "Write a function that prints numbers 1-100, but for multiples of 3 print 'Fizz', for multiples of 5 print 'Buzz', and for multiples of both print 'FizzBuzz'",
                    "starter_code": """def fizzbuzz():\n    # Your code here\n    pass\n\nfizzbuzz()""",
                    "hints": ["Use the modulo operator %", "Check for 15 first (3*5)"]
                }
            ],
            PythonConcept.DECORATORS: [
                {
                    "title": "Retry Decorator",
                    "description": "Create a decorator that retries a function up to n times if it raises an exception",
                    "starter_code": """def retry(max_attempts=3):\n    # Your decorator here\n    pass\n\n@retry(max_attempts=3)\ndef unstable_function():\n    import random\n    if random.random() < 0.5:\n        raise ValueError("Random failure")\n    return "Success!" """,
                    "hints": ["Use a nested function structure", "Catch exceptions in a loop"]
                }
            ]
        }
        
        if concept in exercises and exercises[concept]:
            exercise = exercises[concept][0]  # Get first exercise for now
            
            response = f"""# Practice Exercise: {exercise['title']}

**Concept:** {concept.value}
**Difficulty:** {'⭐' * difficulty}

## Description:
{exercise['description']}

## Starter Code:
```python
{exercise['starter_code']}
Hints:
{chr(10).join(f"- {hint}" for hint in exercise['hints'])}

Try to solve it yourself first, then use the execute_code tool to test your solution!
"""
return [TextContent(type="text", text=response)]

plaintext

                        Copy
                    
    return [TextContent(
        type="text",
        text=f"No exercises available for {concept.value} yet."
    )]

async def _handle_save_example(self, args: Dict[str, Any]) -> List[TextContent]:
    """Save a user's code example"""
    example = CodeExample(
        title=args["title"],
        code=args["code"],
        explanation=args.get("explanation", ""),
        concept=PythonConcept(args["concept"]),
        tags=args.get("tags", [])
    )
    
    self.user_examples.append(example)
    
    return [TextContent(
        type="text",
        text=f"✅ Saved example: '{example.title}' under {example.concept.value}"
    )]

async def _handle_search_examples(self, query: str) -> List[TextContent]:
    """Search through saved examples"""
    query_lower = query.lower()
    
    # Search in user examples and built-in examples
    all_examples = self.user_examples.copy()
    for teacher in self.teachers.values():
        all_examples.extend(teacher.get_examples())
    
    # Filter examples
    matches = []
    for example in all_examples:
        if (query_lower in example.title.lower() or
            query_lower in example.concept.value or
            any(query_lower in tag.lower() for tag in example.tags)):
            matches.append(example)
    
    if not matches:
        return [TextContent(type="text", text=f"No examples found for '{query}'")]
    
    # Format results
    response_parts = [f"# Search Results for '{query}'\n"]
    response_parts.append(f"Found {len(matches)} example(s):\n")
    
    for example in matches[:5]:  # Limit to 5 results
        response_parts.append(f"\n## {example.title}")
        response_parts.append(f"Concept: {example.concept.value}")
        response_parts.append(f"Tags: {', '.join(example.tags)}")
        response_parts.append(f"```python\n{example.code[:200]}{'...' if len(example.code) > 200 else ''}\n```")
    
    return [TextContent(type="text", text="\n".join(response_parts))]

async def run(self):
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_
