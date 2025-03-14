import React, { useState, useEffect } from 'react';

const UserSelector = ({ onUserChange, currentUser }) => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/customers');
        const data = await response.json();
        setUsers(data);
      } catch (error) {
        console.error('Error fetching users:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchUsers();
  }, []);

  if (loading) {
    return (
      <div className="selector-base user-selector">
        Chargement...
      </div>
    );
  }

  return (
    <select
      value={currentUser?.id || ''}
      onChange={(e) => {
        const selectedUser = users.find(u => u.id === e.target.value);
        if (selectedUser) {
          // Ajouter le nom complet à l'objet utilisateur
          const userWithFullName = {
            ...selectedUser,
            fullName: `${selectedUser.first_name} ${selectedUser.last_name}`
          };
          onUserChange(userWithFullName);
        }
      }}
      className="selector-base user-selector"
    >
      <option value="">Sélectionner un utilisateur</option>
      {users.map((user) => (
        <option key={user.id} value={user.id}>
          {user.first_name} {user.last_name}
        </option>
      ))}
    </select>
  );
};

export default UserSelector;