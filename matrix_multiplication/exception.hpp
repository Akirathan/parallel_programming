//
// Created by pal on 17.6.19.
//

#ifndef MATRIX_MULT_EXCEPTION_HPP
#define MATRIX_MULT_EXCEPTION_HPP

#include <exception>
#include <string>
#include <sstream>

/**
 * \brief Specific exception that behaves like a stream, so it can cummulate
 *		error messages more easily.
 */
class StreamException : public std::exception
{
protected:
	std::string mMessage;	///< Internal buffer where the message is kept.

public:
	StreamException() : std::exception() {}
	StreamException(const char *msg) : std::exception(), mMessage(msg) {}
	StreamException(const std::string &msg) : std::exception(), mMessage(msg) {}
	virtual ~StreamException() throw() {}

	virtual const char* what() const throw()
	{
		return mMessage.c_str();
	}

	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	StreamException& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};


/**
 * \brief A stream exception that is base for all runtime errors.
 */
class RuntimeError : public StreamException
{
public:
	RuntimeError() : StreamException() {}
	RuntimeError(const char *msg) : StreamException(msg) {}
	RuntimeError(const std::string &msg) : StreamException(msg) {}
	virtual ~RuntimeError() throw() {}


	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	RuntimeError& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};

#endif //MATRIX_MULT_EXCEPTION_HPP
