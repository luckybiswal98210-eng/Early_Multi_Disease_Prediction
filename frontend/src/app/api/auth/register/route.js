import { NextResponse } from 'next/server';
import connectToDatabase from '@/lib/mongodb';
import User from '@/models/User';
import { mockStore } from '@/lib/mockStore';
import bcrypt from 'bcryptjs';

export async function POST(req) {
  try {
    const { name, email, password, age, gender, contact, role } = await req.json();

    if (!name || !email || !password) {
      return NextResponse.json(
        { error: 'Name, email, and password are required.' },
        { status: 400 }
      );
    }

    const assignedRole = role || (email.toLowerCase().includes('admin') ? 'admin' : 'patient');
    const salt = await bcrypt.genSalt(10);
    const passwordHash = await bcrypt.hash(password, salt);

    const db = await connectToDatabase();

    if (db) {
      // Mongoose/MongoDB Atlas storage
      const existing = await User.findOne({ email: email.toLowerCase() });
      if (existing) {
        return NextResponse.json(
          { error: 'An account with this email already exists.' },
          { status: 409 }
        );
      }

      const newUser = await User.create({
        name,
        email: email.toLowerCase(),
        password: passwordHash,
        role: assignedRole,
        age: Number(age) || 35,
        gender: gender || 'Male',
        contact: contact || '',
      });

      return NextResponse.json({
        success: true,
        user: {
          id: newUser._id.toString(),
          name: newUser.name,
          email: newUser.email,
          role: newUser.role,
          age: newUser.age,
          gender: newUser.gender,
        },
      });
    } else {
      // Fallback mock storage
      const existing = mockStore.findUserByEmail(email);
      if (existing) {
        return NextResponse.json(
          { error: 'An account with this email already exists.' },
          { status: 409 }
        );
      }

      const newUser = mockStore.addUser({
        _id: `usr_${Date.now()}`,
        name,
        email: email.toLowerCase(),
        passwordHash,
        role: assignedRole,
        age: Number(age) || 35,
        gender: gender || 'Male',
        contact: contact || '',
        createdAt: new Date().toISOString(),
      });

      return NextResponse.json({
        success: true,
        user: {
          id: newUser._id,
          name: newUser.name,
          email: newUser.email,
          role: newUser.role,
          age: newUser.age,
          gender: newUser.gender,
        },
      });
    }
  } catch (error) {
    console.error('Registration error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
