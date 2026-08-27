from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=0,
    
)
text="""
Clean Architecture is a software design philosophy that emphasizes the separation of concerns, ensuring that business logic remains independent of frameworks, databases, external APIs, and user interfaces. By organizing code into concentric layers, it creates systems that are testable, maintainable, and adaptable to change over time.
Core Architectural Layers
Entities (Domain Layer): The innermost layer containing core business rules, domain models, and enterprise logic. It has zero knowledge of database schemas, UI frameworks, or third-party libraries.

Use Cases (Application Layer): Contains application-specific business rules. Use cases orchestrate the flow of data to and from entities, defining what actions the system can perform without specifying how data is delivered or stored.

Interface Adapters: Converts data from the format most convenient for use cases and entities to the format most convenient for external agencies like databases, web frameworks, or UI controllers.

Frameworks & Drivers (Infrastructure Layer): The outermost layer consisting of specific tools and technologies—such as database drivers, web servers, ORMs, or messaging queues.

Key Principles
The Dependency Rule: Code dependencies can only point inward. Inner layers know nothing about outer layers. For example, domain logic must never import database implementations or web frameworks.

Database & Framework Independence: The database and UI are treated as external details. You can swap out PostgreSQL for MongoDB, or transition from a REST API to a GraphQL endpoint, without touching the core business logic.

High Testability: Because core business rules are completely decoupled from external frameworks and network dependencies, they can be tested rapidly using plain unit tests without mocking complex infrastructure.

"""

textsplit=text_splitter.split_text(text)
print((textsplit))

